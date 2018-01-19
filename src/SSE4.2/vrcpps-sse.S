#define INSTR vrcpps
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

        vaddps xmm1, xmm0, xmm0     # create 2.0
        vaddps xmm2, xmm0, xmm1     # create 3.0
        vaddps xmm4, xmm1, xmm1     # create 4.0
        vaddps xmm4, xmm4, xmm4     # create 8.0
        vaddps xmm4, xmm4, xmm4     # create 16.0
        vaddps xmm4, xmm4, xmm4     # create 32.0
        vaddps xmm4, xmm4, xmm4     # create 64.0
        vaddps xmm4, xmm4, xmm4     # create 128.0
        vaddps xmm4, xmm4, xmm4     # create 256.0
        vaddps xmm4, xmm4, xmm4     # create 512.0
        vaddps xmm4, xmm4, xmm4     # create 1024.0
        vdivps xmm1, xmm4, xmm2     # create 341.3333
        vdivps xmm2, xmm0, xmm1     # create 1/341.3333
        vaddps xmm0, xmm1, xmm1     # create 2*341.3333
loop:
        inc       i
        INSTR     xmm1, xmm0
        INSTR     xmm2, xmm1
        INSTR     xmm3, xmm2
        cmp       i, N
        INSTR     xmm4, xmm3
        INSTR     xmm5, xmm4
        INSTR     xmm0, xmm5
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
