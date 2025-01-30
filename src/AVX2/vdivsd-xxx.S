#define INSTR vdivsd
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

        vaddsd xmm1, xmm0, xmm0     # create 2.0
        vaddsd xmm2, xmm0, xmm1     # create 3.0
        vaddsd xmm4, xmm1, xmm1     # create 4.0
        vaddsd xmm4, xmm4, xmm4     # create 8.0
        vaddsd xmm4, xmm4, xmm4     # create 16.0
        vaddsd xmm4, xmm4, xmm4     # create 32.0
        vaddsd xmm4, xmm4, xmm4     # create 64.0
        vaddsd xmm4, xmm4, xmm4     # create 128.0
        vaddsd xmm4, xmm4, xmm4     # create 256.0
        vaddsd xmm4, xmm4, xmm4     # create 512.0
        vaddsd xmm4, xmm4, xmm4     # create 1024.0
        vdivsd xmm1, xmm4, xmm2     # create 341.3333
        vdivsd xmm2, xmm0, xmm1     # create 1/341.3333
        vaddsd xmm0, xmm1, xmm1     # create 2*341.3333

        # Mark registers as scalar
        movsd xmm0, xmm0
        movsd xmm1, xmm1
        movsd xmm2, xmm2

loop:
        inc       i
        INSTR     xmm0, xmm0, xmm1
        INSTR     xmm0, xmm0, xmm2
        INSTR     xmm0, xmm0, xmm1
        cmp       i, N
        INSTR     xmm0, xmm0, xmm2
        INSTR     xmm0, xmm0, xmm1
        INSTR     xmm0, xmm0, xmm2
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
