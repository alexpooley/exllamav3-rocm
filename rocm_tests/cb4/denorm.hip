// Does v_dot2_f32_f16 keep fp16 denormal inputs? (default HIP float mode)
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
__global__ void k(const unsigned* a, const unsigned* b, float* o)
{
    int i = threadIdx.x;
    o[i] = __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a[i]), __builtin_bit_cast(hv2, b[i]), 0.0f, false);
}
int main()
{
    unsigned ha[4] = {0x00010001u, 0x00ff0000u, 0x00800080u, 0x64ff0000u};
    unsigned hb[4] = {0x3c003c00u, 0x3c003c00u, 0x4c004c00u, 0x3c003c00u};   // 1, 1, 16, 1
    unsigned *a, *b; float* o; hipMalloc(&a, 16); hipMalloc(&b, 16); hipMalloc(&o, 16);
    hipMemcpy(a, ha, 16, hipMemcpyHostToDevice); hipMemcpy(b, hb, 16, hipMemcpyHostToDevice);
    k<<<1, 4>>>(a, b, o); float r[4]; hipMemcpy(r, o, 16, hipMemcpyDeviceToHost);
    const float d = 1.0f / 16777216.0f;
    printf("2 denormals of code 1 * 1: %g (expect %g)\n", r[0], 2 * d);
    printf("code 255 hi * 1: %g (expect %g)\n", r[1], 255 * d);
    printf("2x code 128 * 16: %g (expect %g)\n", r[2], 2 * 128 * 16 * d);
    printf("normal 0x64ff hi: %g (expect 1279)\n", r[3]);
}
