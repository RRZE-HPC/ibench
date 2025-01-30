#define INSTR vpcmpgtd 
#define NINST 8
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
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (10 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # fill mask register
        mov     eax, 1023
        kmovw   k1, eax
loop:
        inc       i
        INSTR     k1, ymm0, ymm1
        INSTR     k2, ymm1, ymm0
        INSTR     k3, ymm0, ymm2
        INSTR     k4, ymm2, ymm0
        INSTR     k5, ymm1, ymm2
        INSTR     k6, ymm2, ymm1
        INSTR     k7, ymm2, ymm1
        cmp       i, N
        INSTR     k0, ymm2, ymm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
