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
        # create SP 1.0
        vpcmpeqw xmm0, xmm0, xmm0       # all ones
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (11 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # copy SP 1.0
        vmovaps ymm1, ymm0

        # Mark registers AVX
        vmovapd ymm0, ymm0
        vmovapd ymm1, ymm1

loop:
        inc       i
        INSTR     ymm0, ymm1, ymm1
        INSTR     ymm0, ymm1, ymm1
        INSTR     ymm0, ymm1, ymm1
        cmp       i, N
        INSTR     ymm0, ymm1, ymm1
        INSTR     ymm0, ymm1, ymm1
        INSTR     ymm0, ymm1, ymm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
