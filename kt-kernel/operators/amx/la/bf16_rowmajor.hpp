#ifndef KT_AMX_BF16_ROWMAJOR_HPP
#define KT_AMX_BF16_ROWMAJOR_HPP

/**
 * @file bf16_rowmajor.hpp
 * @brief BF16 AMX MoE GEMM kernel that consumes row-major activations and
 *        produces row-major accumulators, removing the BufferA pack and
 *        BufferC unpack passes that GemmKernel224BF16 performs.
 *
 * GemmKernel224BF16 repacks activations into a tile-contiguous BufferA before
 * the GEMM and scatters a tile-blocked BufferC back to row-major afterwards, so
 * every _tile_loadd/_tile_stored sees a 64/128-byte stride. This kernel loads A
 * tiles straight from the row-major activation (lda = row stride in bytes) and
 * stores C tiles straight into a row-major fp32 buffer (ldc = row stride in
 * bytes): BufferA::from_mat degenerates to a copy and BufferC::to_mat to a
 * row-wise fp32 -> bf16 conversion. The two repack waves cost more than the
 * strided tile access they avoid.
 *
 * Weights live in an n-contiguous variant of kt's packed layout -- same VNNI
 * tile-pairs, same [n][k] from_mat source, but the two inner block axes swapped
 * so that tiles adjacent in n are adjacent in memory (see BufferBBF16NContigImpl
 * for the exact index). That layout is not chosen for this kernel's benefit; it
 * is what the decode path requires, and prefill shares the buffer rather than
 * holding a second copy of the weights. The cost of reading it, once the B
 * prefetch below is in place, is 0.6-2.9% at 512 tokens and nothing at 8192.
 *
 * Tile-pair internals and the accumulation order are untouched, so outputs are
 * bit-identical to GemmKernel224BF16 for every shape -- the correctness gate
 * asserts equality, not a tolerance.
 *
 * Three details are load-bearing and were fixed by measurement, not chosen:
 *
 *  - Row padding. For power-of-two k and n the natural row strides (k * 2 and
 *    n * 4 bytes) keep the L1 set index constant across a tile's rows, so all
 *    16 rows of a tile land in one set. Padding each row by 64 bytes shifts the
 *    set index per row and is what makes this kernel win; without it the kernel
 *    loses to GemmKernel224BF16.
 *
 *  - Non-temporal stores on the two write-once paths (the activation copy in
 *    BufferA::from_mat and the output write in BufferC::to_mat). Neither buffer
 *    is read back by this core, so streaming them avoids the read-for-ownership
 *    traffic and the cache pollution a normal store causes.
 *
 *  - A software prefetch of the B panel, one k-step ahead, issued only while
 *    the first m-block of a k-block streams the panel in. This is the same
 *    recipe GemmKernel224BF16 uses; see the comments there for why the gating
 *    and the depth are what they are. Without it this kernel is slower than
 *    GemmKernel224BF16 at small per-expert m.
 *
 * Contract mirrors GemmKernel224BF16: same M/N/K_STEP of 32, same N_BLOCK and
 * K_BLOCK, same split_range_n semantics (nth must equal recommended_nth(n)),
 * m tails handled by padding max_m to M_STEP and discarding the garbage rows in
 * to_mat. n and k must be multiples of 32; there are no tail kernels.
 */

#include <cstring>

#include "amx_kernels.hpp"  // AMX_AVAILABLE, which amx_raw_kernels.hpp expects its includer to provide
#include "amx_raw_kernels.hpp"

namespace amx {

/**
 * @brief Row-major BF16 activation buffer: the packed layout is the source
 *        layout, so from_mat is a copy rather than a tile pack.
 *
 * Rows [m, max_m) are left unwritten and consumed as garbage by the last
 * M_STEP tile, then discarded by BufferC::to_mat -- the same contract as
 * BufferABF16Impl. get_submat returns &a[m_begin][k_begin]; the row stride is
 * k + A_ROW_PAD elements, so the kernel must take lda from the buffer instead
 * of assuming tiles are contiguous.
 */
template <typename K>
struct BufferARowMajorBF16Impl {
  ggml_bf16_t* a;
  int max_m, k;
  size_t stride;  // elements per row, including the pad
  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int ROW_PAD = K::A_ROW_PAD;

  static size_t required_size(int max_m, int k) {
    return sizeof(ggml_bf16_t) * (size_t)max_m * (size_t)(k + ROW_PAD);
  }

  BufferARowMajorBF16Impl(int max_m, int k, void* ptr) : max_m(max_m), k(k), stride((size_t)k + ROW_PAD) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    a = reinterpret_cast<ggml_bf16_t*>(ptr);
  }

  void set_data(void* new_ptr) { a = reinterpret_cast<ggml_bf16_t*>(new_ptr); }

  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    // k is a multiple of K_STEP (32 bf16 = 64 B) and rows are 64 B aligned by
    // construction, so every row is a whole number of cache lines and can be
    // streamed. This buffer is written once and read only by the AMX tile
    // loads, so the stores never need ownership of the lines.
    for (int i = 0; i < m; i++) {
      const char* srow = reinterpret_cast<const char*>(src + (size_t)i * k);
      char* drow = reinterpret_cast<char*>(a + (size_t)i * stride);
      for (int b = 0; b < k * (int)sizeof(ggml_bf16_t); b += 64) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(drow + b), _mm512_loadu_si512(srow + b));
      }
    }
    _mm_sfence();
  }

  ggml_bf16_t* get_submat(int m, int k_total, int m_begin, int k_begin) {
    (void)m;
    (void)k_total;
    return a + (size_t)m_begin * stride + k_begin;
  }
};

/**
 * @brief Row-major FP32 accumulator buffer: no blocked scatter, to_mat is a
 *        row-wise fp32 -> bf16 conversion of this ith's N_BLOCK panel.
 *
 * get_submat returns &c[m_begin][n_begin]; the row stride is n + C_ROW_PAD
 * floats.
 */
template <typename K>
struct BufferCRowMajorFP32Impl {
  float* c;
  int max_m, n;
  size_t stride;  // floats per row, including the pad
  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int ROW_PAD = K::C_ROW_PAD;

  static size_t required_size(int max_m, int n) { return sizeof(float) * (size_t)max_m * (size_t)(n + ROW_PAD); }

  BufferCRowMajorFP32Impl(int max_m, int n, void* ptr) : max_m(max_m), n(n), stride((size_t)n + ROW_PAD) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    c = reinterpret_cast<float*>(ptr);
  }

  void set_data(void* new_ptr) { c = reinterpret_cast<float*>(new_ptr); }

  void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    // One N_STEP of output is 32 bf16 = 64 B, exactly one line. Stream it when
    // the destination row start is line aligned; otherwise fall back to normal
    // stores rather than splitting lines across owners.
    const bool stream = (reinterpret_cast<uintptr_t>(dst) % 64 == 0) && (n % N_STEP == 0);
    for (int i = 0; i < m; i++) {
      for (int j = n_start; j < n_end; j += N_STEP) {
        __m512* lo = reinterpret_cast<__m512*>(c + (size_t)i * stride + j);
        __m512* hi = reinterpret_cast<__m512*>(c + (size_t)i * stride + j + 16);
        ggml_bf16_t* out = dst + (size_t)i * n + j;
        if (stream) {
          _mm512_stream_si512(reinterpret_cast<__m512i*>(out), (__m512i)_mm512_cvtne2ps_pbh(*hi, *lo));
        } else {
          avx512_32xfp32_to_32xbf16(lo, hi, reinterpret_cast<__m512i*>(out));
        }
      }
    }
    if (stream) _mm_sfence();
  }

  float* get_submat(int m, int n_total, int m_begin, int n_begin) {
    (void)m;
    (void)n_total;
    return c + (size_t)m_begin * stride + n_begin;
  }
};

/**
 * @brief BF16 GEMM kernel over row-major A and C, weights shared with
 *        GemmKernel224BF16.
 */
template <typename K>
struct BufferBBF16NContigImpl {
  ggml_bf16_t* b;
  int n, k;
  static constexpr bool SCALE = false;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static constexpr int TILE_N = K::TILE_N;

  static size_t required_size(int n, int k) { return sizeof(ggml_bf16_t) * (size_t)n * k; }

  BufferBBF16NContigImpl(int n, int k, void* ptr) : n(n), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % K_STEP == 0);
    b = reinterpret_cast<ggml_bf16_t*>(ptr);
  }
  void set_data(void* new_ptr) { b = reinterpret_cast<ggml_bf16_t*>(new_ptr); }

  void pack_block(ggml_bf16_t* src, int src_stride, int n_block_begin, int n_block_size) {
    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
        for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
          ggml_bf16_t* d = b + (size_t)n_block_begin * k + (size_t)k_block_begin * n_block_size +
                           (size_t)k_begin * n_block_size + (size_t)n_begin * K_STEP;
          for (int i = 0; i < N_STEP; i++) {
            __m512i* s = (__m512i*)(src + (size_t)(n_begin + i) * src_stride + k_block_begin + k_begin);
            avx512_copy_32xbf16(s, (__m512i*)(d + (size_t)i * K_STEP));
          }
          transpose_16x16_32bit((__m512i*)d);
          transpose_16x16_32bit((__m512i*)(d + TILE_N * K_STEP));
        }
      }
    }
  }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    pack_block(src + (size_t)n_start * k, k, n_start, n_end - n_start);
  }

  void from_mat_strided(ggml_bf16_t* src, int src_stride, int ith, int nth) {
    assert(src_stride >= k);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    pack_block(src + (size_t)n_start * src_stride, src_stride, n_start, n_end - n_start);
  }

  ggml_bf16_t* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    return b + (size_t)n_block_begin * k + (size_t)k_block_begin * n_block_size + (size_t)k_begin * n_block_size +
           (size_t)n_begin * K_STEP;
  }
};

/**
 * @brief Row-major activations and accumulator over an n-contiguous weight
 *        buffer -- the combination to use when the weight layout is fixed by
 *        another consumer and cannot be chosen for the GEMM's convenience.
 *
 * The B prefetch is the one place the weight layout shows through: within a
 * panel the n-contiguous order advances by a whole k-step run per k-step rather
 * than by one tile-pair, so the prefetch strides by that run instead of running
 * contiguously. Everything else is the same recipe GemmKernel224BF16 uses.
 */
struct GemmKernel224BF16RowMajorNC {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 2;
  static constexpr int TILE_M = 16;
  static constexpr int TILE_K = 32;
  static constexpr int TILE_N = 16;
  static constexpr int VNNI_BLK = 2;

  static constexpr int M_STEP = TILE_M * 2;
  static constexpr int N_STEP = TILE_N * 2;
  static constexpr int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 1792;

  static constexpr int A_ROW_PAD = 32;  // bf16 elements
  static constexpr int C_ROW_PAD = 16;  // floats
  static constexpr int B_PREFETCH_K_STEPS = 1;
  static constexpr int B_PREFETCH_LINES = N_STEP * K_STEP * sizeof(ggml_bf16_t) / 64;

  // Weights are NOT in kt's layout. Any code that walks BufferB by hand --
  // notably bf16-moe.hpp's write_weights_to_buffer -- must branch on this.
  static constexpr bool N_CONTIG_B_LAYOUT = true;

  static std::string name() { return "BF16_ROWMAJOR_NC"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    (void)nth;
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() { GemmKernel224BF16::config(); }

  using BufferA = BufferARowMajorBF16Impl<GemmKernel224BF16RowMajorNC>;
  using BufferB = BufferBBF16NContigImpl<GemmKernel224BF16RowMajorNC>;
  using BufferC = BufferCRowMajorFP32Impl<GemmKernel224BF16RowMajorNC>;

  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    const size_t lda = ba->stride * sizeof(dt);
    const size_t ldc = ((size_t)n + C_ROW_PAD) * sizeof(output_t);

    if (k_block_begin == 0) {
      GemmKernel224BF16::clean_c();
    } else {
      GemmKernel224BF16::load_c(c, ldc);
    }

    const bool prefetch_b = m_begin == 0;
    // Elements between one k-step's strip run and the next one.
    const int n_block_size = std::min((int)N_BLOCK, n - n_begin / N_BLOCK * N_BLOCK);
    const size_t k_step_run = (size_t)n_block_size * K_STEP;

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      dt* b = bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      if (prefetch_b) {
        const char* p = reinterpret_cast<const char*>(b + B_PREFETCH_K_STEPS * k_step_run);
        for (int i = 0; i < B_PREFETCH_LINES; i++) {
          _mm_prefetch(p + i * 64, _MM_HINT_T1);
        }
      }
      GemmKernel224BF16::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), lda);
      GemmKernel224BF16::load_b(b, K_STEP * sizeof(dt));
      GemmKernel224BF16::run_tile();
    }

    GemmKernel224BF16::store_c(c, ldc);
  }

  // AVX512-BF16 fallback, used by vec_mul. The B tile-pair is contiguous
  // whichever way the panel is ordered, so this is the same math as
  // GemmKernel224BF16::avx_kernel_4 with A and C rows strided; the layout
  // difference is entirely absorbed by BufferB::get_submat.
  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                           BufferB* bb) {
    const int m_block_end = std::min(m - m_begin, M_STEP);
    const size_t c_row = (size_t)n + C_ROW_PAD;
    const size_t a_row32 = ba->stride / 2;

    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        _mm512_storeu_ps(c + (size_t)m_i * c_row, _mm512_setzero_ps());
        _mm512_storeu_ps(c + (size_t)m_i * c_row + 16, _mm512_setzero_ps());
      }
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      const int32_t* a32 = reinterpret_cast<const int32_t*>(ba->get_submat(m, k, m_begin, k_block_begin + k_begin));
      __m512bh* b512 = reinterpret_cast<__m512bh*>(bb->get_submat(n, k, n_begin, k_block_begin + k_begin));

      for (int k_i = 0; k_i < 16; k_i += 4) {
        __m512bh b0_lo = b512[k_i], b0_hi = b512[16 + k_i];
        __m512bh b1_lo = b512[k_i + 1], b1_hi = b512[16 + k_i + 1];
        __m512bh b2_lo = b512[k_i + 2], b2_hi = b512[16 + k_i + 2];
        __m512bh b3_lo = b512[k_i + 3], b3_hi = b512[16 + k_i + 3];

        for (int m_i = 0; m_i < m_block_end; m_i++) {
          const int32_t* a_row = a32 + (size_t)m_i * a_row32;
          float* c_row_ptr = c + (size_t)m_i * c_row;

          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(a_row[k_i]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(a_row[k_i + 1]);
          __m512bh ma2 = (__m512bh)_mm512_set1_epi32(a_row[k_i + 2]);
          __m512bh ma3 = (__m512bh)_mm512_set1_epi32(a_row[k_i + 3]);

          __m512 acc0 = _mm512_loadu_ps(c_row_ptr);
          __m512 acc1 = _mm512_loadu_ps(c_row_ptr + 16);
          acc0 = _mm512_dpbf16_ps(acc0, ma0, b0_lo);
          acc1 = _mm512_dpbf16_ps(acc1, ma0, b0_hi);
          acc0 = _mm512_dpbf16_ps(acc0, ma1, b1_lo);
          acc1 = _mm512_dpbf16_ps(acc1, ma1, b1_hi);
          acc0 = _mm512_dpbf16_ps(acc0, ma2, b2_lo);
          acc1 = _mm512_dpbf16_ps(acc1, ma2, b2_hi);
          acc0 = _mm512_dpbf16_ps(acc0, ma3, b3_lo);
          acc1 = _mm512_dpbf16_ps(acc1, ma3, b3_hi);
          _mm512_storeu_ps(c_row_ptr, acc0);
          _mm512_storeu_ps(c_row_ptr + 16, acc1);
        }
      }
    }
  }
};

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF16RowMajorNC::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF16RowMajorNC::BufferB> bb,
                    std::shared_ptr<GemmKernel224BF16RowMajorNC::BufferC> bc, int ith, int nth) {
  float_mat_vec<GemmKernel224BF16RowMajorNC, true>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF16RowMajorNC::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF16RowMajorNC::BufferB> bb,
                    std::shared_ptr<GemmKernel224BF16RowMajorNC::BufferC> bc, int ith, int nth) {
  float_mat_vec<GemmKernel224BF16RowMajorNC, false>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

#endif  // KT_AMX_BF16_ROWMAJOR_HPP
