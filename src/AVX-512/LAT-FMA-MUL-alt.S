#define INSTR vfmadd213sd
#define NINST 1
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
        # copy DP 1.0
        movsd xmm0, xmm0
        movsd xmm1, xmm0
        movsd xmm2, xmm0

loop:
        inc       i
        vfmadd231sd     xmm0, xmm1, xmm2
        vmulsd          xmm0, xmm0, xmm1
        cmp       i, N
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
