// tile_k2 게이트 1·2: plan == dense / 마스크 전량 1에서 avx_kernel_4와 비트일치.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "la/amx_kernels.hpp"
#include "la/amx_raw_kernels.hpp"
#include "la/bf16_rowmajor.hpp"
#include "la/tile_k2.hpp"

using namespace amx;
using K = GemmKernelTileK2BF16;

// 작은 정수만 쓰므로 상위 16비트 절단이 무손실이다 (ggml 링크 회피).
static ggml_bf16_t f2b(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  ggml_bf16_t b;
  b.bits = (uint16_t)(u >> 16);
  return b;
}

int run_case(int n, int k, bool use_mask) {
  std::mt19937 g(1234 + n + k);
  std::uniform_int_distribution<int> d(-4, 4);
  std::vector<ggml_bf16_t> x(k), w((size_t)n * k);
  for (auto& v : x) v = f2b((float)d(g));
  for (auto& v : w) v = f2b((float)d(g));

  void* bp = std::aligned_alloc(64, K::BufferB::required_size(n, k));
  K::BufferB bb(n, k, bp);
  // from_mat은 split_range_n을 쓰므로 **호출당 N_BLOCK 하나**만 채운다 (nth는 무시된다).
  // 한 번만 부르면 나머지 열이 미터치 상태로 남아 공유 zero page를 읽고, 그러면
  // 이 비교가 "둘 다 0"이라 통과해 버린다 — 꼬리 검증이 통째로 무의미해진다.
  for (int t = 0; t * K::N_BLOCK < n; t++) bb.from_mat(w.data(), t, 1);
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(K::M_STEP, k));
  K::BufferA ba(K::M_STEP, k, ap);
  ba.from_mat(1, x.data(), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc(K::M_STEP, n, cp);
  void* cp2 = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc2(K::M_STEP, n, cp2);

  // 레퍼런스: 상속받은 avx_kernel_4 전 K (k_block 단위)
  std::memset(bc2.c, 0, K::BufferC::required_size(K::M_STEP, n));
  for (int kb = 0; kb < k; kb += K::K_BLOCK)
    for (int nb = 0; nb < n; nb += K::N_STEP)
      K::avx_kernel_4(1, n, k, 0, nb, kb, bc2.get_submat(1, n, 0, nb), &ba, &bb);


  std::vector<uint16_t> mask(k / 32, 0xFFFF);
  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  // n을 넘겨야 부분 블록용 오프셋 표가 만들어진다 (n % N_BLOCK == 0이면 무시된다).
  K::build_plan(use_mask ? mask.data() : nullptr, x.data(), k, arena.data(), &plan, n);
  K::gemv_slab(1, n, k, 0, n, &ba, &bb, &bc, &plan);

  int bad = 0;
  for (int j = 0; j < n; j++)
    if (std::memcmp(&bc.c[j], &bc2.c[j], sizeof(float)) != 0) bad++;
  std::printf("tile_k2 n=%4d k=%4d %-6s: bitwise=%s (%d mismatched)\n", n, k,
              use_mask ? "mask1" : "dense", bad == 0 ? "yes" : "NO", bad);
  std::free(bp); std::free(ap); std::free(cp); std::free(cp2);
  return bad;
}

int main() {
  int bad = 0;
  for (int k : {64, 256, 2048}) { bad += run_case(256, k, false); bad += run_case(256, k, true); }
  bad += run_case(768, 2048, false);
  bad += run_case(512, 1792 + 256, true);

  // 게이트 3: 부분 n-블록. N_BLOCK(256)의 배수가 아닌 n에서도 상속받은
  // avx_kernel_4와 비트일치해야 한다 — 꼬리 블록은 타일 컬럼 stride가 다르므로
  // plan의 koff_tail을 타고, 그 표가 틀리면 남의 열을 읽어 값이 어긋난다.
  std::printf("-- 부분 블록 (n %% 256 != 0) --\n");
  for (int n : {128, 288, 384, 640, 1056}) {  // 전부 32의 배수, 256의 배수는 아님
    bad += run_case(n, 2048, false);
    bad += run_case(n, 2048, true);
  }
  // 실제로 재려던 치수: down = 640행 x 2560열의 전치 쪽(n=640, k=640)
  bad += run_case(640, 640, true);
  std::printf(bad ? "tile_k2 gate FAILED\n" : "tile_k2 gate OK\n");
  return bad != 0;
}
