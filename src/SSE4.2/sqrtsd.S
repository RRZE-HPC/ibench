#define INSTR sqrtsd
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

        vaddpd xmm1, xmm0, xmm0     # create 2.0
        vaddpd xmm2, xmm0, xmm1     # create 3.0
        vaddpd xmm4, xmm1, xmm1     # create 4.0
        vaddpd xmm4, xmm4, xmm4     # create 8.0
        vaddpd xmm4, xmm4, xmm4     # create 16.0
        vaddpd xmm4, xmm4, xmm4     # create 32.0
        vaddpd xmm4, xmm4, xmm4     # create 64.0
        vaddpd xmm4, xmm4, xmm4     # create 128.0
        vaddpd xmm4, xmm4, xmm4     # create 256.0
        vaddpd xmm4, xmm4, xmm4     # create 512.0
        vaddpd xmm4, xmm4, xmm4     # create 1024.0
        vdivpd xmm1, xmm4, xmm2     # create 341.3333
        vdivpd xmm2, xmm0, xmm1     # create 1/341.3333
        vaddpd xmm0, xmm1, xmm1     # create 2*341.3333
        vmovapd xmm10, xmm0          # save value
loop:
        inc       i
        INSTR     xmm1, xmm0
        addsd    xmm1, xmm10
        INSTR     xmm2, xmm1
        addsd    xmm2, xmm10
        INSTR     xmm3, xmm2
        addsd    xmm3, xmm10
        cmp       i, N
        INSTR     xmm4, xmm3
        addsd    xmm4, xmm10
        INSTR     xmm5, xmm4
        addsd    xmm5, xmm10
        INSTR     xmm0, xmm5
        addsd    xmm0, xmm10
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
