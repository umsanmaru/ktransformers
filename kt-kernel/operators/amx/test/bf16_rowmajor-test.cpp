/**
 * @brief Correctness gate for GemmKernel224BF16RowMajorNC against GemmKernel224BF16.
 *
 * Both kernels are driven over their full path -- BufferB::from_mat for the
 * weights, BufferA::from_mat for the activations, mat_mul for every ith, then
 * BufferC::to_mat -- on the same row-major [m][k] activation and [n][k] weight.
 * The weight layouts differ -- one n-contiguous, one kt's own -- but the tile
 * pairs handed to AMX and the order they are accumulated in are identical, so
 * the requirement is bit-identical output, not a tolerance. A double-precision reference is
 * also checked so that a shared bug in both kernels cannot pass silently.
 *
 * Shapes cover the contract edges: m tails inside and across M_STEP, an n tail
 * block (n % N_BLOCK != 0, which exercises split_range_n panels), and k beyond
 * K_BLOCK so the k-block tail path runs.
 *
 * Build (from kt-kernel/operators/amx/test):
 *   g++ -O2 -std=c++20 -mavx512f -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni \
 *       -mamx-tile -mamx-bf16 -mamx-int8 -DHAVE_AMX \
 *       -I.. -I../../.. -I../../../.. -I../../../../third_party \
 *       -I../../../../third_party/llama.cpp \
 *       bf16_rowmajor-test.cpp -o bf16_rowmajor-test
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "../la/bf16_rowmajor.hpp"

namespace {

using amx::GemmKernel224BF16;
using amx::GemmKernel224BF16RowMajorNC;

float bf16_to_f32(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

uint16_t f32_to_bf16(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

void* alloc_aligned(size_t bytes, std::vector<char>& owner) {
  owner.assign(bytes + 64, 0);
  return (void*)(((uintptr_t)owner.data() + 63) & ~(uintptr_t)63);
}

// Run one kernel end to end and return its [m][n] bf16 output.
template <typename K>
std::vector<uint16_t> run_kernel(int m, int n, int k, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  const int nth = K::recommended_nth(n);
  const int max_m = (m + K::M_STEP - 1) / K::M_STEP * K::M_STEP;

  std::vector<char> bmem, amem, cmem;
  auto bb = std::make_shared<typename K::BufferB>(n, k, alloc_aligned(K::BufferB::required_size(n, k), bmem));
  for (int ith = 0; ith < nth; ith++) bb->from_mat((ggml_bf16_t*)b.data(), ith, nth);

  auto ba = std::make_shared<typename K::BufferA>(max_m, k, alloc_aligned(K::BufferA::required_size(max_m, k), amem));
  auto bc = std::make_shared<typename K::BufferC>(max_m, n, alloc_aligned(K::BufferC::required_size(max_m, n), cmem));
  ba->from_mat(m, (ggml_bf16_t*)a.data(), 0, 1);

  K::config();
  for (int ith = 0; ith < nth; ith++) amx::mat_mul(m, n, k, ba, bb, bc, ith, nth);

  std::vector<uint16_t> out((size_t)m * n, 0);
  for (int ith = 0; ith < nth; ith++) bc->to_mat(m, (ggml_bf16_t*)out.data(), ith, nth);
  return out;
}

bool run_case(int m, int n, int k, unsigned seed) {
  std::mt19937 g(seed);
  // Small integers are exactly representable in bf16, so the double reference
  // differs from the kernels only by accumulation effects.
  auto rb = [&] { return f32_to_bf16((float)((int)(g() % 9) - 4)); };

  std::vector<uint16_t> a((size_t)m * k), b((size_t)n * k);
  for (auto& x : a) x = rb();
  for (auto& x : b) x = rb();

  std::vector<uint16_t> out_kt = run_kernel<GemmKernel224BF16>(m, n, k, a, b);
  std::vector<uint16_t> out_rm = run_kernel<GemmKernel224BF16RowMajorNC>(m, n, k, a, b);

  size_t mismatched = 0;
  double max_rel = 0.0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      size_t idx = (size_t)i * n + j;
      if (out_kt[idx] != out_rm[idx]) mismatched++;

      double ref = 0.0;
      for (int kk = 0; kk < k; kk++) {
        ref += (double)bf16_to_f32(a[(size_t)i * k + kk]) * (double)bf16_to_f32(b[(size_t)j * k + kk]);
      }
      double got = (double)bf16_to_f32(out_rm[idx]);
      double denom = std::fabs(ref) > 1.0 ? std::fabs(ref) : 1.0;
      max_rel = std::max(max_rel, std::fabs(got - ref) / denom);
    }
  }

  const bool ok = mismatched == 0 && max_rel < 1e-2;
  std::printf("rowmajor+NC m=%4d n=%4d k=%4d: bitwise=%s (%zu mismatched) rel=%.2e %s\n", m, n, k,
              mismatched == 0 ? "yes" : "NO", mismatched, max_rel, ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool passed = true;
  passed = run_case(32, 256, 256, 1) && passed;      // one m-block, one n-block
  passed = run_case(17, 256, 256, 2) && passed;      // m tail inside M_STEP
  passed = run_case(96, 512, 256, 3) && passed;      // several m-blocks, two n-blocks
  passed = run_case(65, 288, 256, 4) && passed;      // n tail block (288 % 256 != 0)
  passed = run_case(64, 256, 2048, 5) && passed;     // k > K_BLOCK, k-block tail
  passed = run_case(128, 256, 4096, 6) && passed;    // several k-blocks
  std::printf("bf16_rowmajor-test %s\n", passed ? "OK" : "FAILED");
  return passed ? 0 : 1;
}
