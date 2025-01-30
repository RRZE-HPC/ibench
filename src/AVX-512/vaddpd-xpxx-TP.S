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
        INSTR     xmm3{k1}, xmm0, xmm1
        INSTR     xmm4{k1}, xmm1, xmm0
        INSTR     xmm5{k1}, xmm0, xmm2
        INSTR     xmm6{k1}, xmm2, xmm0
        INSTR     xmm7{k1}, xmm1, xmm2
        INSTR     xmm8{k1}, xmm2, xmm1
        INSTR     xmm9{k1}, xmm2, xmm1
        cmp       i, N
        INSTR     xmm10{k1}, xmm2, xmm1
        INSTR     xmm11{k1}, xmm2, xmm1
        INSTR     xmm12{k1}, xmm2, xmm1
        INSTR     xmm13{k1}, xmm2, xmm1
        INSTR     xmm14{k1}, xmm2, xmm1
        INSTR     xmm15{k1}, xmm2, xmm1
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
