#define INSTR vmulsd
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
        vpslld xmm0, xmm0, 25           # logical left shift: 11111110..0 (25 = 32 - (8 - 1))
        vpsrld xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # create SP 2.0
        vaddps xmm1, xmm0, xmm0
        # create SP 0.5
        vdivps xmm2, xmm0, xmm1
loop:
        inc       i
        INSTR     xmm3, xmm0, xmm1
        INSTR     xmm4, xmm1, xmm0
        INSTR     xmm5, xmm0, xmm2
        cmp       i, N
        INSTR     xmm6, xmm2, xmm0
        INSTR     xmm7, xmm1, xmm2
        INSTR     xmm8, xmm2, xmm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
