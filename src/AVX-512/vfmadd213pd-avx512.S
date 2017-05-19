#define INSTR vfmadd213pd
#define NINST 6
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
        # create SSE DP 1.0
        vpcmpeqw xmm0, xmm0, xmm0       # all ones
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (11 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # expand from AVX to AVX-512
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1
        # copy DP 1.0
        vmovapd zmm1, zmm0
loop:
        inc       i
        INSTR     zmm0, zmm1, zmm1
        INSTR     zmm0, zmm1, zmm1
        INSTR     zmm0, zmm1, zmm1
        cmp       i, N
        INSTR     zmm0, zmm1, zmm1
        INSTR     zmm0, zmm1, zmm1
        INSTR     zmm0, zmm1, zmm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
