#define INSTR vmovapd
#define NINST 4
#define N edi
#define i r8d


.intel_syntax noprefix
.globl ninst
.data
ninst:
.long NINST
.align 64
PI:
.long  0xf01b866e,0x400921f9, 0xf01b866e,0x400921f9, 0xf01b866e,0x400921f9, 0xf01b866e,0x400921f9
.text
.globl latency
.type latency, @function
.align 64
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

        # Mark registers AVX-512
        vmovapd zmm0, zmm0
        vmovapd zmm1, zmm1

loop:
        inc       i
        INSTR     zmm0, [rip+PI]
        INSTR     zmm1, [rip+PI]
        INSTR     ymm2, [rip+PI]
        INSTR     zmm3, [rip+PI]
        INSTR     zmm4, [rip+PI]
        INSTR     ymm5, [rip+PI]
        INSTR     zmm6, [rip+PI]
        INSTR     zmm7, [rip+PI]
        INSTR     ymm8, [rip+PI]
        INSTR     zmm9, [rip+PI]
        INSTR     zmm10, [rip+PI]
        INSTR     ymm11, [rip+PI]
        cmp       i, N
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
