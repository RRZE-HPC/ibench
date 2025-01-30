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
        vmovapd zmm0, zmm0
        vmovapd zmm1, zmm1
        vmovapd zmm2, zmm2
        vmovapd zmm3, zmm3
        vmovapd zmm4, zmm4
        vmovapd zmm5, zmm5
        vmovapd zmm6, zmm6
        vmovapd zmm7, zmm7
        vmovapd zmm8, zmm8
        vmovapd zmm9, zmm8
        vmovapd zmm10, zmm8
        vmovapd zmm11, zmm8
        vmovapd zmm12, zmm8
        vmovapd zmm13, zmm8
        vmovapd zmm14, zmm8
        vmovapd zmm15, zmm8

loop:
        inc       i
        INSTR     zmm3, zmm0
        INSTR     zmm4, zmm1
        INSTR     zmm5, zmm2
        INSTR     zmm6, zmm0
        INSTR     zmm7, zmm1
        INSTR     zmm8, zmm2
        INSTR     zmm9, zmm0
        INSTR     zmm10, zmm1
        INSTR     zmm11, zmm2
        INSTR     zmm12, zmm0
        INSTR     zmm13, zmm1
        INSTR     zmm14, zmm0
        INSTR     zmm15, zmm1

        INSTR     zmm3, zmm0
        INSTR     zmm4, zmm1
        INSTR     zmm5, zmm2
        INSTR     zmm6, zmm0
        INSTR     zmm7, zmm1
        INSTR     zmm8, zmm2
        INSTR     zmm9, zmm0
        INSTR     zmm10, zmm1
        INSTR     zmm11, zmm2
        INSTR     zmm12, zmm0
        INSTR     zmm13, zmm1
        INSTR     zmm14, zmm0
        INSTR     zmm15, zmm1
        cmp       i, N
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
