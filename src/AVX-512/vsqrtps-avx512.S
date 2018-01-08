#define INSTR vsqrtps
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
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1

        vaddps zmm1, zmm0, zmm0     # create 2.0
        vaddps zmm2, zmm0, zmm1     # create 3.0
        vaddps zmm4, zmm1, zmm1     # create 4.0
        vaddps zmm4, zmm4, zmm4     # create 8.0
        vaddps zmm4, zmm4, zmm4     # create 16.0
        vaddps zmm4, zmm4, zmm4     # create 32.0
        vaddps zmm4, zmm4, zmm4     # create 64.0
        vaddps zmm4, zmm4, zmm4     # create 128.0
        vaddps zmm4, zmm4, zmm4     # create 256.0
        vaddps zmm4, zmm4, zmm4     # create 512.0
        vaddps zmm4, zmm4, zmm4     # create 1024.0
        vdivps zmm1, zmm4, zmm2     # create 341.3333
        vdivps zmm2, zmm0, zmm1     # create 1/341.3333
        vaddps zmm0, zmm1, zmm1     # create 2*341.3333
        vmovaps zmm10, zmm0          # save value
loop:
        inc       i
        INSTR     zmm1, zmm0
        vaddps    zmm1, zmm1, zmm10
        INSTR     zmm2, zmm1
        vaddps    zmm2, zmm2, zmm10
        INSTR     zmm3, zmm2
        vaddps    zmm3, zmm3, zmm10
        cmp       i, N
        INSTR     zmm4, zmm3
        vaddps    zmm4, zmm4, zmm10
        INSTR     zmm5, zmm4
        vaddps    zmm5, zmm5, zmm10
        INSTR     zmm0, zmm5
        vaddps    zmm0, zmm0, zmm10
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
