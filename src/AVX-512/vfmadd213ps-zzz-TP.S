#define INSTR vfmadd213ps
#define NINST 13
#define N edi
#define i r8d


.intel_syntax noprefix
.globl ninst
.data
ninst:
.long NINST
.text
.globl latency
.type latency, @function
.align 32
latency:
        push      rbp
        mov       rbp, rsp
        xor       i, i
        test      N, N
        jle       done
        # create SSE SP 1.0
        vpcmpeqw xmm0, xmm0, xmm0       # all ones
        vpslld xmm0, xmm0, 25           # logical left shift: 11111110..0 (25 = 32 - (8 - 1))
        vpsrld xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # expand from AVX to AVX-512
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1
        # copy SP 1.0
        vmovaps zmm1, zmm0
loop:
        inc       i
        INSTR     zmm3, zmm0, zmm1
        INSTR     zmm4, zmm1, zmm0
        INSTR     zmm5, zmm0, zmm2
        INSTR     zmm6, zmm2, zmm0
        INSTR     zmm7, zmm1, zmm2
        INSTR     zmm8, zmm2, zmm1
        INSTR     zmm9, zmm2, zmm1
        cmp       i, N
        INSTR     zmm10, zmm2, zmm1
        INSTR     zmm11, zmm2, zmm1
        INSTR     zmm12, zmm2, zmm1
        INSTR     zmm13, zmm2, zmm1
        INSTR     zmm14, zmm2, zmm1
        INSTR     zmm15, zmm2, zmm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
