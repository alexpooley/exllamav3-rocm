// Persistent ("megakernel") prototype for one or two gated-MLP blocks of an EXL3 dense model on RDNA3
// (gfx1100). One launch runs: fused pre-norm residual add + RMSNorm + Hadamard input transform
// (rms_norm_kernel<RES_IN> + tail) -> gate/up matmul with the silu*up epilogue -> down matmul (fp32 out),
// for 1 or 2 layers in a row. Workgroups pull work units from a global queue in dependency order and
// wait only on the counters of the units they really read from. The GEMV unit is the port's
// exl3_rdna3_unit (same decode, dot and reduction code, so the output is bit-identical for the same
// split configuration), with a hook that waits for the input after the weight prefetch ring is issued.
//
// Counters are monotonic across launches: the host passes the launch number (epoch, from 1) and every
// wait target is epoch * (units per launch). The queue grows by total + grid per launch (each workgroup
// makes exactly one failing pull), so no reset kernel is needed.

#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/HIPContext.h>
#include "quant/exl3_rdna3_kernel_hip.cuh"

namespace pm {

using namespace exl3_rdna3_ns;

struct Layer
{
    const float* xin;               // [M][D] fp32 sublayer output, added to the residual
    float* resid;                   // [M][D] fp32 residual, updated in place
    const __hip_bfloat16* nw;       // norm weight
    const uint64_t* gu_suh_tab;     // gate / up suh
    uint2* xh_gu;                   // 2 slabs
    float* xcs_gu;
    Exl3Rdna3MTab mt;               // gate / up tables + act epilogue
    half* c_gu;                     // [2][M][I]
    float* ws_gu;
    int* cnt_gu;
    const uint32_t* b_d;
    const half* svh_d;
    float* c_d;                     // [M][D]
    float* ws_d;
    int* cnt_d;
    int* dep;                       // [0] norm rows done, [1] down units done, [2 + s] gate/up units done for down split s
    float eps, cbias, cscale;
};

struct Params
{
    Layer L[2];
    int nlayers;
    int M, D, I;
    int s_gu, ks_gu, s_d, ks_d;
    unsigned* queue;
    unsigned epoch;
    int mode;                       // experiments: 1 no waits, 2 skip down units, 4 skip norm units
    int order;                      // gate/up unit order: 0 = mgemm grid order, 1 = group-major
    int prefetch;                   // 1: L2-prefetch the unit's weights while its input is not ready
    unsigned* dummy;
    unsigned long long* trace;      // optional: per unit [start, input ready, done, workgroup] (steady counter)
};

__device__ __forceinline__ bool ready(int* c, int target)
{
    return (int) (__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - target) >= 0;
}

// Thread 0 spins on the counter (agent-scope atomic loads bypass L0 / GL1), then every thread
// invalidates its L0 / GL1 so the data loads that follow see the producers' writes
__device__ int g_nowait_dummy;
__device__ __forceinline__ void wait_ge(int* c, int target)
{
    if (threadIdx.x == 0)
        while ((int) (__hip_atomic_load(c, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) - target) < 0)
            __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ __forceinline__ void signal(int* c)
{
#ifdef PM_FULL_FENCE
    __threadfence();             // seq_cst agent fence: also invalidates L0 / GL1 (GL1 is shared by the shader array)
#else
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // stores done and visible at L2, no invalidate
#endif
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_fetch_add(c, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// Touch one dword per 128-byte line of the unit's weights (slices [ks0, ks1), 8 tiles at tile nt0) so
// they sit in L2 / MALL when the main loop streams them
template <int TWORDS>
__device__ __forceinline__ void l2_prefetch(const uint32_t* B, int n_stride, int nt0, int ks0, int ks1, unsigned* dummy)
{
    constexpr int LINES = 8 * TWORDS * 4 / 128;   // lines per slice for the unit's 8 tiles
    const size_t sstride = (size_t) (n_stride / 16) * TWORDS;
    uint32_t acc = 0;
    const int n = (ks1 - ks0) * LINES;
    for (int i = threadIdx.x; i < n; i += 256)
    {
        const int s = ks0 + i / LINES;
        const int l = i % LINES;
        acc += B[s * sstride + (size_t) nt0 * TWORDS + l * 32];
    }
    if (acc == 0x9e3779b9u) *dummy = acc;
}

__device__ __forceinline__ uint64_t uniform_ptr(uint64_t p)
{
    const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) p);
    const uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t) (p >> 32));
    return ((uint64_t) hi << 32) | lo;
}

// exl3_rdna3_unit (exl3_rdna3_kernel_hip.cuh), RAW path only, with the shared memory passed in and
// wait() called once the first weight prefetch ring is in flight, before any read of xh / xcs
template <int bits, int MR, bool c_fp32, typename W>
#ifdef PM_UNIT_NOINLINE
__device__ __attribute__((noinline)) void unit
#else
__device__ __forceinline__ void unit
#endif
(
    const int e, const int rc, const int group, const int split, const int row_chunks,
    const uint2* __restrict__ xh, const uint32_t* __restrict__ B, void* __restrict__ C,
    int size_m, int size_k, int size_n, int* __restrict__ counters, const float* __restrict__ xcs,
    float* __restrict__ ws, const half* __restrict__ svh, int splits, int ks_per_split,
    const Exl3Rdna3MTab& mt,
    uint2* xs, float (*red)[128], float* xsum, int& last_flag,
    W wait
)
{
    constexpr bool half_k = false;
    constexpr int cb = 2;
    int n_stride = size_n;
    #define PM_U(x) x = (decltype(x)) uniform_ptr((uint64_t) (x))
    PM_U(xh); PM_U(B); PM_U(C); PM_U(counters); PM_U(xcs); PM_U(ws); PM_U(svh);
    size_m = __builtin_amdgcn_readfirstlane(size_m); size_k = __builtin_amdgcn_readfirstlane(size_k);
    size_n = __builtin_amdgcn_readfirstlane(size_n); splits = __builtin_amdgcn_readfirstlane(splits);
    ks_per_split = __builtin_amdgcn_readfirstlane(ks_per_split);
    if (mt.b)
    {
        const int src = mt.src ? mt.src[e] : e;
        // Table loads inside the persistent loop come back in VGPRs (vector loads: the loop also stores);
        // without readfirstlane the buffer resource is rebuilt with v_readfirstlane in the hot loop (32 per
        // k-slice step), which cost 25% at M = 8
        B = (const uint32_t*) uniform_ptr(mt.b[e]);
        svh = (const half*) uniform_ptr(mt.svh[e]);
        if (mt.c) C = (void*) mt.c[e];
        else C = (void*) ((char*) C + (size_t) e * size_m * size_n * (c_fp32 ? 4 : 2));
        if (mt.n_stride) n_stride = mt.n_stride[e];
        xh += (size_t) src * size_m * (size_k / 16) * 4;
        xcs += (size_t) src * size_m * (size_k / 128);
        ws += (size_t) e * splits * size_m * size_n;
        counters += (size_t) e * row_chunks * (size_n / 128);
    }

    constexpr int TWORDS = 8 * bits;
    constexpr bool DIRECT = cb == 2 && !half_k && bits >= 2 && bits <= 6 && (bits != 4 || MR >= 4);
    constexpr int LPT = DIRECT ? (bits >= 5 ? 4 : 2) : (TWORDS + 31) / 32;
    constexpr int KCS = XS_BYTES / (MR * 2) / 16 / 8 * 8;
    constexpr int PF = 4;
    constexpr int NACC = 2;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q = lane & 3;

    const int ntiles = size_n / 16;
    const int groups = ntiles / NT;
    const int row0 = rc * MR;
    const int rows = min(MR, size_m - row0);
    const int kslices = size_k / 16;
    const int ks_begin = split * ks_per_split;
    const int ks_end = min(kslices, ks_begin + ks_per_split);
    constexpr int TPW = EXL3_RDNA3_TPW;
    constexpr int WAVES = NT / TPW;
    const int nt0 = group * NT + warp;
    const size_t sstride = (size_t) (n_stride / 16) * TWORDS;

    const __amdgpu_buffer_rsrc_t brsrc = __builtin_amdgcn_make_buffer_rsrc((void*) B, (short) 0, 0x7fffffff, 0x31004000);

    float acc[MR][TPW][NACC];
    #pragma unroll
    for (int m = 0; m < MR; ++m)
        #pragma unroll
        for (int j = 0; j < TPW; ++j)
            #pragma unroll
            for (int a = 0; a < NACC; ++a) acc[m][j][a] = 0.0f;

    bool first = true;
#ifdef PM_KC_NOUNROLL
    #pragma clang loop unroll(disable)
#endif
    for (int kc0 = ks_begin; kc0 < ks_end; kc0 += KCS)
    {
        const int nk = min(ks_end - kc0, KCS);

        uint32_t pf[PF][TPW][LPT];
        int lofs[LPT];
        #pragma unroll
        for (int l = 0; l < LPT; ++l) lofs[l] = min(l * 32 + lane, TWORDS - 1);
        if constexpr (DIRECT)
        {
            int c = 0;
            uint32_t unused[8];
            auto record = [&] (int jw) -> uint32_t { lofs[c++] = jw; return 0u; };
            windows8<bits>(record, lane, unused);
        }

        uint32_t voff[TPW][LPT];
        #pragma unroll
        for (int j = 0; j < TPW; ++j)
            #pragma unroll
            for (int l = 0; l < LPT; ++l) voff[j][l] = (uint32_t) ((nt0 + j * WAVES) * TWORDS + lofs[l]) * 4u;
        const uint32_t sstride_b = (uint32_t) sstride * 4u;
        auto ldw = [&] (int slice, int j, int l) -> uint32_t
        {
            const uint32_t soff = (uint32_t) slice * sstride_b;
            return __builtin_amdgcn_raw_buffer_load_b32(brsrc, voff[j][l], soff, 6);
        };

        #pragma unroll
        for (int d = 0; d < PF; ++d)
            #pragma unroll
            for (int j = 0; j < TPW; ++j)
                #pragma unroll
                for (int l = 0; l < LPT; ++l)
                    pf[d][j][l] = ldw(kc0 + d, j, l);

        __syncthreads();
        if (first)
        {
            first = false;
            wait(B, n_stride, group * NT, ks_begin, ks_end);
            if (threadIdx.x < MR)
            {
                float s = 0.0f;
                if ((int) threadIdx.x < rows)
                    for (int c = ks_begin / 8; c < ks_end / 8; ++c)
                        s += xcs[(size_t) (row0 + threadIdx.x) * (kslices / 8) + c];
                xsum[threadIdx.x] = s;
            }
        }
        {
            const int per_row = nk * 2;
            for (int idx = threadIdx.x; idx < MR * per_row; idx += THREADS)
            {
                const int m = idx / per_row;
                const int j = idx % per_row;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m < rows)
                    v = ((const uint4*) (xh + ((size_t) (row0 + m) * kslices + kc0) * 4))[j];
                ((uint4*) (xs + (size_t) m * KCS * 4))[j] = v;
            }
        }
        __syncthreads();

#ifdef PM_IB_NOUNROLL
        #pragma clang loop unroll(disable)
#endif
        for (int ib = 0; ib < nk; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                const int i = ib + d;
                uint32_t wd[TPW][4];
                #pragma unroll
                for (int j = 0; j < TPW; ++j)
                {
                    uint32_t w[LPT];
                    #pragma unroll
                    for (int l = 0; l < LPT; ++l) w[l] = pf[d][j][l];
                    const int inext = min(i + PF, nk - 1);
                    auto refill = [&] ()
                    {
                        #pragma unroll
                        for (int l = 0; l < LPT; ++l)
                            pf[d][j][l] = ldw(kc0 + inext, j, l);
                        __builtin_amdgcn_sched_barrier(0);
                    };
                    int wc = 0;
                    auto gather = [&] (int jj) -> uint32_t
                    {
                        if constexpr (DIRECT) return w[wc++];
                        else if constexpr (LPT == 1) return __shfl(w[0], jj);
                        else
                        {
                            uint32_t lo = __shfl(w[0], jj & 31);
                            uint32_t hi = __shfl(w[1], jj & 31);
                            return jj < 32 ? lo : hi;
                        }
                    };
                    uint32_t win[8];
                    windows8<bits>(gather, lane, win);
                    refill();
                    wd[j][0] = mul1_raw_pair(win[0], win[1]);
                    wd[j][1] = mul1_raw_pair(win[2], win[3]);
                    wd[j][2] = mul1_raw_pair(win[4], win[5]);
                    wd[j][3] = mul1_raw_pair(win[6], win[7]);
                }
                #pragma unroll
                for (int m = 0; m < MR; ++m)
                {
                    const uint2 xv = xs[(m * KCS + i) * 4 + q];
                    #pragma unroll
                    for (int j = 0; j < TPW; ++j)
                    {
                        acc[m][j][0] = dot2(wd[j][0], xv.x, acc[m][j][0]);
                        acc[m][j][NACC - 2] = dot2(wd[j][1], xv.y, acc[m][j][NACC - 2]);
                        acc[m][j][1] = dot2(wd[j][2], xv.x, acc[m][j][1]);
                        acc[m][j][NACC - 1] = dot2(wd[j][3], xv.y, acc[m][j][NACC - 1]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MR; ++m)
    #pragma unroll
    for (int j = 0; j < TPW; ++j)
    {
        #pragma unroll
        for (int f = 0; f < 2; ++f)
        {
            float v = acc[m][j][f];
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 2);
            const float k_inv = __half2float(__ushort_as_half(0x1eee));
            const float k_bias = __half2float(__ushort_as_half(0xc931));
            v = fmaf(v, k_inv, k_bias * xsum[m]);
            acc[m][j][f] = v;
        }
        if (q == 0)
        {
            const int tl = warp + j * WAVES;
            red[m][tl * 16 + (lane >> 2)] = acc[m][j][0];
            red[m][tl * 16 + 8 + (lane >> 2)] = acc[m][j][1];
        }
    }
    __syncthreads();

    if (splits > 1)
    {
        const int gidx = rc * groups + group;
        float* wp = ws + ((size_t) split * size_m + row0) * size_n + group * 128;
        for (int idx = threadIdx.x; idx < rows * 128; idx += THREADS)
        {
            const int m = idx >> 7, c = idx & 127;
            wp[(size_t) m * size_n + c] = red[m][c];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int prev = atomicAdd(&counters[gidx], 1);
            last_flag = prev == splits - 1;
        }
        __syncthreads();
        if (!last_flag) return;
        __threadfence();

        for (int idx = threadIdx.x; idx < rows * 128; idx += THREADS)
        {
            const int m = idx >> 7, c = idx & 127;
            const float* rp = ws + (size_t) (row0 + m) * size_n + group * 128 + c;
            float s = 0.0f;
            for (int sp = 0; sp < splits; ++sp)
                s += __builtin_nontemporal_load(rp + (size_t) sp * size_m * size_n);
            red[m][c] = s;
        }
        if (threadIdx.x == 0) counters[gidx] = 0;
        __syncthreads();
    }

    for (int m = warp; m < rows; m += WAVES)
    {
        float h0 = red[m][lane * 4 + 0];
        float h1 = red[m][lane * 4 + 1];
        float h2 = red[m][lane * 4 + 2];
        float h3 = red[m][lane * 4 + 3];
        had128(h0, h1, h2, h3, lane);
        const int col = group * 128 + lane * 4;
        const half2* sp = (const half2*) (svh + col);
        const half2 s01 = sp[0], s23 = sp[1];
        const float r = 0.088388347648f;
        h0 *= r * __low2float(s01);
        h1 *= r * __high2float(s01);
        h2 *= r * __low2float(s23);
        h3 *= r * __high2float(s23);
        const size_t o = (size_t) (row0 + m) * n_stride + col;
        if constexpr (c_fp32)
        {
            *((float4*) (((float*) C) + o)) = make_float4(h0, h1, h2, h3);
        }
        else
        {
            half2* cp = (half2*) (((half*) C) + o);
            cp[0] = __floats2half2_rn(h0, h1);
            cp[1] = __floats2half2_rn(h2, h3);
        }
    }

    if constexpr (!c_fp32)
    {
        if (mt.act_g)
        {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                int* cnt = mt.act_cnt + rc * groups + group;
                const int prev = __hip_atomic_fetch_add(cnt, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
                last_flag = prev == 1;
                if (prev == 1) __hip_atomic_store(cnt, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            __syncthreads();
            if (last_flag)
            {
                __threadfence();
                auto sigmoid2 = [] (half2 x) -> half2
                {
                    const half2 one = __float2half2_rn(1.0f);
                    return h2rcp(__hadd2(one, h2exp(__hneg2(x))));
                };
                for (int m = warp; m < rows; m += WAVES)
                {
                    const size_t o = (size_t) (row0 + m) * mt.act_k + group * 128 + lane * 4;
                    const half2* gp = (const half2*) (mt.act_g + o);
                    const half2* up = (const half2*) (mt.act_u + o);
                    half2 x01 = gp[0], x23 = gp[1];
                    x01 = __hmul2(__hmul2(x01, sigmoid2(x01)), up[0]);
                    x23 = __hmul2(__hmul2(x23, sigmoid2(x23)), up[1]);
                    exl3_rdna3_had::transform_block(x01, x23, mt.act_suh, mt.act_xh, mt.act_xcs, row0 + m, group, mt.act_k, lane);
                }
            }
        }
    }
}

// rms_norm_kernel<RES_IN, float, half, bf16, float> for one row, 1024 virtual threads emulated by 256
// (same per-thread column order and the same two-level xor butterfly, so the result is bit-identical),
// then its RDNA3 input-transform tail for the two gate / up sources. y is staged in LDS (ys)
__device__ __forceinline__ void norm_row(const Layer& L, int row, int M, int D, half* ys, float* sums)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int columns = D / 4;
    const float4* x = (const float4*) (L.xin + (size_t) row * D);
    float4* r = (float4*) (L.resid + (size_t) row * D);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int vw = warp + 8 * j;
        const int vt = vw * 32 + lane;
        float sum = 0.0f;
        for (int column = vt; column < columns; column += 1024)
        {
            float4 x4 = x[column];
            const float4 r4 = r[column];
            x4.x += r4.x; x4.y += r4.y; x4.z += r4.z; x4.w += r4.w;
            r[column] = x4;
            sum = fmaf(x4.x, x4.x, sum);
            sum = fmaf(x4.y, x4.y, sum);
            sum = fmaf(x4.z, x4.z, sum);
            sum = fmaf(x4.w, x4.w, sum);
        }
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset);
        if (lane == 0) sums[vw] = sum;
    }
    __threadfence_block();
    __syncthreads();
    float sum = sums[lane];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset);
    const float rmf = rsqrtf(sum / (float) D + L.eps) * L.cscale;

    for (int column = threadIdx.x; column < columns; column += 256)
    {
        float4 x4 = r[column];
        const uint2 wr = ((const uint2*) L.nw)[column];
        float4 w4;
        w4.x = __uint_as_float(wr.x << 16);
        w4.y = __uint_as_float(wr.x & 0xffff0000u);
        w4.z = __uint_as_float(wr.y << 16);
        w4.w = __uint_as_float(wr.y & 0xffff0000u);
        if (L.cbias != 0.0f) { w4.x += L.cbias; w4.y += L.cbias; w4.z += L.cbias; w4.w += L.cbias; }
        x4.x = x4.x * w4.x * rmf;
        x4.y = x4.y * w4.y * rmf;
        x4.z = x4.z * w4.z * rmf;
        x4.w = x4.w * w4.w * rmf;
        half2* yp = (half2*) (ys + column * 4);
        yp[0] = __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y));
        yp[1] = __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w));
    }
    __syncthreads();
    const int kblocks = D / 128;
    for (int task = warp; task < kblocks * 2; task += 8)
    {
        const int src = task / kblocks;
        const int c = task % kblocks;
        const half2* yp = (const half2*) (ys + c * 128 + lane * 4);
        exl3_rdna3_had::transform_block
        (
            yp[0], yp[1], (const half*) L.gu_suh_tab[src],
            L.xh_gu + (size_t) src * M * (D / 16) * 4,
            L.xcs_gu + (size_t) src * M * kblocks,
            row, c, D, lane
        );
    }
}

template <int bits, int MR>
__global__ __launch_bounds__(256)
void persist_mlp_kernel(Params p)
{
    constexpr int KCS = XS_BYTES / (MR * 2) / 16 / 8 * 8;
    __shared__ uint2 xs[MR * KCS * 4];
    __shared__ float red[MR < 2 ? 2 : MR][128];
    __shared__ float xsum[MR];
    __shared__ int last_flag;
    __shared__ unsigned s_unit;

    const int M = p.M, D = p.D, I = p.I;
    const int G_gu = I / 128;
    const int G_d = D / 128;
    const int kb_d = p.ks_d / 8;
    const unsigned n_gu = G_gu * 2 * p.s_gu;
    const unsigned n_d = G_d * p.s_d;
    const unsigned T = M + n_gu + n_d;
    const unsigned total = T * p.nlayers;
    const unsigned base = (p.epoch - 1u) * (total + gridDim.x);
    const int ep = (int) p.epoch;

#if defined(PM_PRIO) && PM_PRIO == 1
    // Wave arbitration is oldest-first: in a persistent grid the late-placed workgroups starve for the
    // whole launch. Give them the higher priority instead
    switch ((blockIdx.x * 4) / gridDim.x)
    {
        case 0: break;
        case 1: __builtin_amdgcn_s_setprio(1); break;
        case 2: __builtin_amdgcn_s_setprio(2); break;
        default: __builtin_amdgcn_s_setprio(3); break;
    }
#endif

    while (true)
    {
        __syncthreads();
        if (threadIdx.x == 0) s_unit = atomicAdd(p.queue, 1u) - base;
        __syncthreads();
        unsigned u = __builtin_amdgcn_readfirstlane(s_unit);   // uniform: keeps B / rsrc in SGPRs (else waterfall loops)
        const unsigned ug = u;
        if (p.trace && threadIdx.x == 0 && u < total) { p.trace[4 * u] = __builtin_readsteadycounter(); p.trace[4 * u + 1] = 0; p.trace[4 * u + 3] = blockIdx.x; }
        if (u >= total) break;
        const int li = u / T;
        u -= li * T;
        const Layer& L = p.L[li];

        if (p.mode & 4) { if (u < (unsigned) M) continue; }
        if ((p.mode & 2) && u >= M + n_gu) continue;
        if (u < (unsigned) M)
        {
            if (li > 0) wait_ge(p.L[li - 1].dep + 1, ep * (int) n_d);
            norm_row(L, u, M, D, (half*) xs, &red[0][0]);
            signal(L.dep + 0);
            if (p.trace && threadIdx.x == 0) p.trace[4 * ug + 2] = __builtin_readsteadycounter();
        }
        else if (u < M + n_gu)
        {
            u -= M;
            // Launch order of the port's mgemm grid (x = group + groups * split, z = entry): concurrent units
            // then read whole contiguous k-slice rows of one matrix (DRAM page locality); g-major order
            // (both matrices, both splits of each group together) ran at half the bandwidth
            int e, s, g;
            if (p.order == 0) { g = u % G_gu; s = (u / G_gu) % p.s_gu; e = u / (G_gu * p.s_gu); }
            else { e = u & 1; s = (u >> 1) % p.s_gu; g = (u >> 1) / p.s_gu; }
            auto w = [&] (const uint32_t* B, int ns, int nt0, int k0, int k1)
            {
                if (p.prefetch && !ready(L.dep + 0, ep * M)) l2_prefetch<8 * bits>(B, ns, nt0, k0, k1, p.dummy);
                if (!(p.mode & 1)) wait_ge(L.dep + 0, ep * M);
                if (p.trace && threadIdx.x == 0) p.trace[4 * ug + 1] = __builtin_readsteadycounter();
            };
            unit<bits, MR, false>(e, 0, g, s, 1, L.xh_gu, nullptr, L.c_gu, M, D, I, L.cnt_gu, L.xcs_gu, L.ws_gu,
                                  nullptr, p.s_gu, p.ks_gu, L.mt, xs, red, xsum, last_flag, w);
            signal(L.dep + 2 + g / kb_d);
            if (p.trace && threadIdx.x == 0) p.trace[4 * ug + 2] = __builtin_readsteadycounter();
        }
        else
        {
            u -= M + n_gu;
            const int s = u / G_d;
            const int g = u % G_d;
            const int g0 = s * kb_d;
            const int g1 = min(G_gu, g0 + kb_d);
            const int target = ep * (g1 - g0) * 2 * p.s_gu;
            int* c = L.dep + 2 + s;
            auto w = [&] (const uint32_t* B, int ns, int nt0, int k0, int k1)
            {
                if (p.prefetch && !ready(c, target)) l2_prefetch<8 * bits>(B, ns, nt0, k0, k1, p.dummy);
                if (!(p.mode & 1)) wait_ge(c, target);
                if (p.trace && threadIdx.x == 0) p.trace[4 * ug + 1] = __builtin_readsteadycounter();
            };
            unit<bits, MR, true>(0, 0, g, s, 1, L.mt.act_xh, L.b_d, L.c_d, M, I, D, L.cnt_d, L.mt.act_xcs, L.ws_d,
                                 L.svh_d, p.s_d, p.ks_d, Exl3Rdna3MTab {}, xs, red, xsum, last_flag, w);
            signal(L.dep + 1);
            if (p.trace && threadIdx.x == 0) p.trace[4 * ug + 2] = __builtin_readsteadycounter();
        }
    }
}

// A/B control: one matmul phase as a plain grid (one unit per workgroup, blockIdx order, no waits), using
// the same unit code. Inputs must already be in place (run the persistent kernel once first)
template <int bits, int MR, int PART>
__global__ __launch_bounds__(256)
void part_kernel(Params p)
{
    constexpr int KCS = XS_BYTES / (MR * 2) / 16 / 8 * 8;
    __shared__ uint2 xs[MR * KCS * 4];
    __shared__ float red[MR < 2 ? 2 : MR][128];
    __shared__ float xsum[MR];
    __shared__ int last_flag;
    const Layer& L = p.L[0];
    const int G_gu = p.I / 128, G_d = p.D / 128;
    const int u = blockIdx.x;
    auto w = [&] (const uint32_t*, int, int, int, int) {};
    if constexpr (PART == 1)
    {
        const int g = u % G_gu, s = (u / G_gu) % p.s_gu, e = u / (G_gu * p.s_gu);
        unit<bits, MR, false>(e, 0, g, s, 1, L.xh_gu, nullptr, L.c_gu, p.M, p.D, p.I, L.cnt_gu, L.xcs_gu, L.ws_gu,
                              nullptr, p.s_gu, p.ks_gu, L.mt, xs, red, xsum, last_flag, w);
    }
    else
    {
        const int s = u / G_d, g = u % G_d;
        unit<bits, MR, true>(0, 0, g, s, 1, L.mt.act_xh, L.b_d, L.c_d, p.M, p.I, p.D, L.cnt_d, L.mt.act_xcs, L.ws_d,
                             L.svh_d, p.s_d, p.ks_d, Exl3Rdna3MTab {}, xs, red, xsum, last_flag, w);
    }
}

// A/B control: gate/up units only, pulled from a queue by a persistent grid, in a kernel that contains
// nothing else (separates the codegen of the combined kernel from the pull pattern). q[0] queue,
// q[1] exit count; the last workgroup out resets both
template <int bits, int MR>
__global__ __launch_bounds__(256)
void pgu_kernel(Params p, unsigned* q)
{
    constexpr int KCS = XS_BYTES / (MR * 2) / 16 / 8 * 8;
    __shared__ uint2 xs[MR * KCS * 4];
    __shared__ float red[MR < 2 ? 2 : MR][128];
    __shared__ float xsum[MR];
    __shared__ int last_flag;
    __shared__ unsigned s_unit;
    const Layer& L = p.L[0];
    const int G_gu = p.I / 128;
    const unsigned n1 = G_gu * 2 * p.s_gu;
    const unsigned n = n1 * (p.mode > 1 ? p.mode : 1);   // mode = R: the unit list R times (timing only)
    auto w = [&] (const uint32_t*, int, int, int, int) {};
    while (true)
    {
        __syncthreads();
        if (threadIdx.x == 0) s_unit = atomicAdd(q, 1u);
        __syncthreads();
        unsigned u = __builtin_amdgcn_readfirstlane(s_unit);
        if (u >= n) break;
        u %= n1;
        const int g = u % G_gu, s = (u / G_gu) % p.s_gu, e = u / (G_gu * p.s_gu);
        unit<bits, MR, false>(e, 0, g, s, 1, L.xh_gu, nullptr, L.c_gu, p.M, p.D, p.I, L.cnt_gu, L.xcs_gu, L.ws_gu,
                              nullptr, p.s_gu, p.ks_gu, L.mt, xs, red, xsum, last_flag, w);
    }
    if (threadIdx.x == 0 && atomicAdd(q + 1, 1u) == gridDim.x - 1) { q[0] = 0; q[1] = 0; }
}

}  // namespace pm

// Layer pointer list order (int64): xin, resid, nw, gu_suh_tab, xh_gu, xcs_gu, b_tab, svh_tab, c_gu, ws_gu,
// cnt_gu, act_suh, act_xh, act_xcs, act_cnt, b_d, svh_d, c_d, ws_d, cnt_d, dep
// floats: eps, cbias, cscale per layer
// cfg: M, D, I, s_gu, ks_gu, s_d, ks_d, queue, epoch, grid, prefetch, nlayers, dummy
static pm::Layer make_layer(const std::vector<int64_t>& v, const std::vector<double>& f, int M, int I)
{
    TORCH_CHECK(v.size() == 21 && f.size() == 3, "bad layer list");
    pm::Layer L {};
    auto P = [&] (int i) { return (void*) v[i]; };
    L.xin = (const float*) P(0);
    L.resid = (float*) P(1);
    L.nw = (const __hip_bfloat16*) P(2);
    L.gu_suh_tab = (const uint64_t*) P(3);
    L.xh_gu = (uint2*) P(4);
    L.xcs_gu = (float*) P(5);
    L.mt = Exl3Rdna3MTab {};
    L.mt.b = (const uint64_t*) P(6);
    L.mt.svh = (const uint64_t*) P(7);
    L.c_gu = (half*) P(8);
    L.ws_gu = (float*) P(9);
    L.cnt_gu = (int*) P(10);
    L.mt.act_g = (const half*) P(8);
    L.mt.act_u = ((const half*) P(8)) + (size_t) M * I;
    L.mt.act_suh = (const half*) P(11);
    L.mt.act_xh = (uint2*) P(12);
    L.mt.act_xcs = (float*) P(13);
    L.mt.act_cnt = (int*) P(14);
    L.mt.act_k = I;
    L.b_d = (const uint32_t*) P(15);
    L.svh_d = (const half*) P(16);
    L.c_d = (float*) P(17);
    L.ws_d = (float*) P(18);
    L.cnt_d = (int*) P(19);
    L.dep = (int*) P(20);
    L.eps = (float) f[0];
    L.cbias = (float) f[1];
    L.cscale = (float) f[2];
    return L;
}

void run(std::vector<int64_t> l0, std::vector<double> f0, std::vector<int64_t> l1, std::vector<double> f1,
         std::vector<int64_t> cfg)
{
    pm::Params p {};
    p.M = (int) cfg[0]; p.D = (int) cfg[1]; p.I = (int) cfg[2];
    p.s_gu = (int) cfg[3]; p.ks_gu = (int) cfg[4]; p.s_d = (int) cfg[5]; p.ks_d = (int) cfg[6];
    p.queue = (unsigned*) cfg[7]; p.epoch = (unsigned) cfg[8];
    const int grid = (int) cfg[9];
    p.prefetch = (int) cfg[10];
    p.nlayers = (int) cfg[11];
    p.dummy = (unsigned*) cfg[12];
    p.order = cfg.size() > 15 ? (int) cfg[15] : 0;
    p.trace = cfg.size() > 14 ? (unsigned long long*) cfg[14] : nullptr;
    const int bits = cfg.size() > 13 ? (int) cfg[13] : 4;
    p.L[0] = make_layer(l0, f0, p.M, p.I);
    if (p.nlayers > 1) p.L[1] = make_layer(l1, f1, p.M, p.I);
    TORCH_CHECK(bits == 4, "bits 4 only");
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    p.mode = cfg.size() > 17 ? (int) cfg[17] : 0;
    const int partsel = cfg.size() > 16 ? (int) cfg[16] : 0;
    if (partsel)
    {
        if (partsel == 3)
        {
            unsigned* q = (unsigned*) cfg[18];
            if (p.M == 1) hipLaunchKernelGGL((pm::pgu_kernel<4, 1>), dim3(grid), dim3(256), 0, stream, p, q);
            else hipLaunchKernelGGL((pm::pgu_kernel<4, 8>), dim3(grid), dim3(256), 0, stream, p, q);
            return;
        }
        const int n = partsel == 1 ? p.I / 128 * 2 * p.s_gu : p.D / 128 * p.s_d;
        if (p.M == 1) { if (partsel == 1) hipLaunchKernelGGL((pm::part_kernel<4, 1, 1>), dim3(n), dim3(256), 0, stream, p);
                        else hipLaunchKernelGGL((pm::part_kernel<4, 1, 2>), dim3(n), dim3(256), 0, stream, p); }
        else { if (partsel == 1) hipLaunchKernelGGL((pm::part_kernel<4, 8, 1>), dim3(n), dim3(256), 0, stream, p);
               else hipLaunchKernelGGL((pm::part_kernel<4, 8, 2>), dim3(n), dim3(256), 0, stream, p); }
        return;
    }
    if (p.M == 1)
        hipLaunchKernelGGL((pm::persist_mlp_kernel<4, 1>), dim3(grid), dim3(256), 0, stream, p);
    else if (p.M <= 8)
        hipLaunchKernelGGL((pm::persist_mlp_kernel<4, 8>), dim3(grid), dim3(256), 0, stream, p);
    else TORCH_CHECK(false, "M <= 8 only");
}

int occupancy(int M)
{
    int n = 0;
    if (M == 1) hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, pm::persist_mlp_kernel<4, 1>, 256, 0);
    else hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, pm::persist_mlp_kernel<4, 8>, 256, 0);
    return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run);
    m.def("occupancy", &occupancy);
}
