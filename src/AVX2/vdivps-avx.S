#define INSTR vdivps
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
        vinsertf128 ymm0, ymm0, xmm0, 0x1

        vaddps ymm1, ymm0, ymm0     # create 2.0
        vaddps ymm2, ymm0, ymm1     # create 3.0
        vaddps ymm4, ymm1, ymm1     # create 4.0
        vaddps ymm4, ymm4, ymm4     # create 8.0
        vaddps ymm4, ymm4, ymm4     # create 16.0
        vaddps ymm4, ymm4, ymm4     # create 32.0
        vaddps ymm4, ymm4, ymm4     # create 64.0
        vaddps ymm4, ymm4, ymm4     # create 128.0
        vaddps ymm4, ymm4, ymm4     # create 256.0
        vaddps ymm4, ymm4, ymm4     # create 512.0
        vaddps ymm4, ymm4, ymm4     # create 1024.0
        vdivps ymm1, ymm4, ymm2     # create 341.3333
        vdivps ymm2, ymm0, ymm1     # create 1/341.3333
        vaddps ymm0, ymm1, ymm1     # create 2*341.3333
loop:
        inc       i
        INSTR     ymm0, ymm0, ymm1
        INSTR     ymm0, ymm0, ymm2
        INSTR     ymm0, ymm0, ymm1
        cmp       i, N
        INSTR     ymm0, ymm0, ymm2
        INSTR     ymm0, ymm0, ymm1
        INSTR     ymm0, ymm0, ymm2
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
