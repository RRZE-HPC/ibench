#define INSTR vdivpd
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
        vinsertf128 ymm0, ymm0, xmm0, 0x1

        vaddpd ymm1, ymm0, ymm0     # create 2.0
        vaddpd ymm2, ymm0, ymm1     # create 3.0
        vaddpd ymm4, ymm1, ymm1     # create 4.0
        vaddpd ymm4, ymm4, ymm4     # create 8.0
        vaddpd ymm4, ymm4, ymm4     # create 16.0
        vaddpd ymm4, ymm4, ymm4     # create 32.0
        vaddpd ymm4, ymm4, ymm4     # create 64.0
        vaddpd ymm4, ymm4, ymm4     # create 128.0
        vaddpd ymm4, ymm4, ymm4     # create 256.0
        vaddpd ymm4, ymm4, ymm4     # create 512.0
        vaddpd ymm4, ymm4, ymm4     # create 1024.0
        vdivpd ymm1, ymm4, ymm2     # create 341.3333
        vdivpd ymm2, ymm0, ymm1     # create 1/341.3333
        vaddpd ymm0, ymm1, ymm1     # create 2*341.3333
loop:
        inc       i
        INSTR     ymm3, ymm0, ymm1
        INSTR     ymm4, ymm1, ymm0
        INSTR     ymm5, ymm0, ymm2
        cmp       i, N
        INSTR     ymm6, ymm2, ymm0
        INSTR     ymm7, ymm1, ymm2
        INSTR     ymm8, ymm2, ymm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
