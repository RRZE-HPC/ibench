#define INSTR vmovapd
#define NINST 26
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
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (10 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # copy SP 1.0
        vmovapd xmm1, xmm0
        vmovapd xmm2, xmm0

        # Mark registers as AVX
        vmovapd ymm0, ymm0
        vmovapd ymm1, ymm1
        vmovapd ymm2, ymm2
        vmovapd ymm3, ymm3
        vmovapd ymm4, ymm4
        vmovapd ymm5, ymm5
        vmovapd ymm6, ymm6
        vmovapd ymm7, ymm7
        vmovapd ymm8, ymm8
        vmovapd ymm9, ymm8
        vmovapd ymm10, ymm8
        vmovapd ymm11, ymm8
        vmovapd ymm12, ymm8
        vmovapd ymm13, ymm8
        vmovapd ymm14, ymm8
        vmovapd ymm15, ymm8

loop:
        inc       i
        INSTR     ymm3, ymm0
        INSTR     ymm4, ymm1
        INSTR     ymm5, ymm2
        INSTR     ymm6, ymm0
        INSTR     ymm7, ymm1
        INSTR     ymm8, ymm2
        INSTR     ymm9, ymm0
        INSTR     ymm10, ymm1
        INSTR     ymm11, ymm2
        INSTR     ymm12, ymm0
        INSTR     ymm13, ymm1
        INSTR     ymm14, ymm0
        INSTR     ymm15, ymm1

        INSTR     ymm3, ymm0
        INSTR     ymm4, ymm1
        INSTR     ymm5, ymm2
        INSTR     ymm6, ymm0
        INSTR     ymm7, ymm1
        INSTR     ymm8, ymm2
        INSTR     ymm9, ymm0
        INSTR     ymm10, ymm1
        INSTR     ymm11, ymm2
        INSTR     ymm12, ymm0
        INSTR     ymm13, ymm1
        INSTR     ymm14, ymm0
        INSTR     ymm15, ymm1
        cmp       i, N
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
