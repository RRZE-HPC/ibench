#define INSTR vtestpd
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

        # Mark registers as SSE
        movapd xmm0, xmm0
        movapd xmm1, xmm1
        movapd xmm2, xmm2
        movapd xmm3, xmm3
        movapd xmm4, xmm4
        movapd xmm5, xmm5
        movapd xmm6, xmm6
        movapd xmm7, xmm7
        movapd xmm8, xmm8
        movapd xmm9, xmm8
        movapd xmm10, xmm8
        movapd xmm11, xmm8
        movapd xmm12, xmm8
        movapd xmm13, xmm8
        movapd xmm14, xmm8
        movapd xmm15, xmm8

loop:
        inc       i
        INSTR     xmm3, xmm0
        INSTR     xmm4, xmm1
        INSTR     xmm5, xmm2
        INSTR     xmm6, xmm0
        INSTR     xmm7, xmm1
        INSTR     xmm8, xmm2
        INSTR     xmm9, xmm0
        INSTR     xmm10, xmm1
        INSTR     xmm11, xmm2
        INSTR     xmm12, xmm0
        INSTR     xmm13, xmm1
        INSTR     xmm14, xmm0
        INSTR     xmm15, xmm1

        INSTR     xmm3, xmm0
        INSTR     xmm4, xmm1
        INSTR     xmm5, xmm2
        INSTR     xmm6, xmm0
        INSTR     xmm7, xmm1
        INSTR     xmm8, xmm2
        INSTR     xmm9, xmm0
        INSTR     xmm10, xmm1
        INSTR     xmm11, xmm2
        INSTR     xmm12, xmm0
        INSTR     xmm13, xmm1
        INSTR     xmm14, xmm0
        INSTR     xmm15, xmm1
        cmp       i, N
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
