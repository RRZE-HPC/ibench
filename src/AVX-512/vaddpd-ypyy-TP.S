#define INSTR vaddpd
#define NINST 13
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
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (10 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # expand from SSE to AVX
        vinsertf128 ymm0, ymm0, xmm0, 0x1
        # expand from AVX to AVX-512
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1
        # copy DP 1.0
        vmovapd zmm1, zmm0
        # fill mask register
        mov     eax, 1023
        kmovw   k1, eax
loop:
        inc       i
        INSTR     ymm3{k1}, ymm0, ymm1
        INSTR     ymm4{k1}, ymm1, ymm0
        INSTR     ymm5{k1}, ymm0, ymm2
        INSTR     ymm6{k1}, ymm2, ymm0
        INSTR     ymm7{k1}, ymm1, ymm2
        INSTR     ymm8{k1}, ymm2, ymm1
        INSTR     ymm9{k1}, ymm2, ymm1
        cmp       i, N
        INSTR     ymm10{k1}, ymm2, ymm1
        INSTR     ymm11{k1}, ymm2, ymm1
        INSTR     ymm12{k1}, ymm2, ymm1
        INSTR     ymm13{k1}, ymm2, ymm1
        INSTR     ymm14{k1}, ymm2, ymm1
        INSTR     ymm15{k1}, ymm2, ymm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
