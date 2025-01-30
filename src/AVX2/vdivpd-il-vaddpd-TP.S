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
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (10 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # copy SP 1.0
        vmovapd ymm1, ymm0
        vmovapd ymm2, ymm0

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
loop:
        inc       i
        INSTR     ymm3, ymm0, ymm1
        vaddpd    ymm9, ymm0, ymm1
        INSTR     ymm4, ymm1, ymm0
        vaddpd    ymm10, ymm0, ymm1
        INSTR     ymm5, ymm0, ymm2
        vaddpd    ymm11, ymm0, ymm1
        cmp       i, N
        INSTR     ymm6, ymm2, ymm0
        vaddpd    ymm12, ymm0, ymm1
        INSTR     ymm7, ymm1, ymm2
        vaddpd    ymm13, ymm0, ymm1
        INSTR     ymm8, ymm2, ymm1
        vaddpd    ymm14, ymm0, ymm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
