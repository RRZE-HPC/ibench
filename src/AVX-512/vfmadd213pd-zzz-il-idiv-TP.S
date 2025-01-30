#define INSTR vfmadd213pd
#define NINST 5
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
        # create SSE DP 1.0
        vpcmpeqw xmm0, xmm0, xmm0       # all ones
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (11 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # expand from AVX to AVX-512
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1
        # copy DP 1.0
        vmovapd zmm1, zmm0
        push    r15
        push    r14
        push    r12
        push    r11
        push    r10
        push    r9
        push    rdx
        push    rcx
        push    rbx
        push    rax
loop:
        inc       i
        INSTR     zmm3, zmm0, zmm1
        INSTR     zmm8, zmm0, zmm1
        shrd      r15, r14, 8
        INSTR     zmm4, zmm1, zmm0
        INSTR     zmm9, zmm0, zmm1
        shrd      r12, r11, 8
        INSTR     zmm5, zmm0, zmm2
        INSTR     zmm10, zmm0, zmm1
        shrd      r10, r9, 8
        INSTR     zmm6, zmm2, zmm0
        INSTR     zmm11, zmm0, zmm1
        shrd      rdx, rcx, 8
        INSTR     zmm7, zmm1, zmm2
        INSTR     zmm12, zmm0, zmm1
        shrd      rbx, rax, 8
        cmp       i, N
        jl        loop
done:
        pop rax
        pop rbx
        pop rcx
        pop rdx
        pop r9
        pop r10
        pop r11
        pop r12
        pop r14
        pop r15
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
