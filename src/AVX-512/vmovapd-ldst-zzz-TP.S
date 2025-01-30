#define INSTR vmovapd
#define NINST 6
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
PI2:
.long  0xd0158667,0x444921f0, 0xf01b4666,0x606921f6, 0x001b860e,0x123921f9, 0xf678866e,0x44562164
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
        INSTR     [rip+PI], zmm0
        INSTR     zmm6, [rip+PI]
        INSTR     [rip+PI], zmm1
        INSTR     zmm7, [rip+PI]
        INSTR     [rip+PI], zmm2
        INSTR     zmm8, [rip+PI]
        cmp       i, N
        INSTR     [rip+PI], zmm3
        INSTR     zmm9, [rip+PI]
        INSTR     [rip+PI], zmm4
        INSTR     zmm10, [rip+PI]
        INSTR     [rip+PI], zmm5
        INSTR     zmm11, [rip+PI]
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
