#define INSTR vdivpd
#define NINST 1
#define N edi
#define i r8d


.intel_syntax noprefix
.globl ninst
.data
ninst:
.long NINST
.align 32
PI:
.long  0xf01b866e,0x400921f9, 0xf01b866e,0x400921f9, 0xf01b866e,0x400921f9, 0xf01b866e,0x400921f9
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
        push r12
        push r13
        push r14
        push r15
loop:
        inc       i
        INSTR     ymm2, ymm0, ymm1
        vcvtsi2ss  xmm3, xmm0, r9
        vcvtsi2ss  xmm4, xmm1, r10
        vcvtsi2ss  xmm5, xmm0, r11
        vcvtsi2ss  xmm11, xmm0, r13
        vmovupd    [rip+PI], ymm1
        vmovupd    [rip+PI], ymm0
        vmovupd    [rip+PI], ymm1
        vmovupd    [rip+PI], ymm0
        #vcvtsi2ss    xmm9, xmm0, r12
        cmp       i, N
        #INSTR     ymm10, ymm1, ymm0
        #vcvtsi2ss    xmm12, xmm1, r14
        #vcvtsi2ss    xmm13, xmm0, r15
        jl        loop
done:
        pop r15
        pop r14
        pop r13
        pop r12
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
