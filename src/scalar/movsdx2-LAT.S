#define INSTR mulsd
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
        vpsllq xmm0, xmm0, 54           # logical left shift: 11111110..0 (54 = 64 - (11 - 1))
        vpsrlq xmm0, xmm0, 2            # logical right shift: 1 bit for sign; leading mantissa bit is zero
        # create SP 2.0
        vaddpd xmm1, xmm0, xmm0
        # create SP 0.5
        vdivpd xmm2, xmm0, xmm1

        # Mark registers as scalar
        movsd xmm0, xmm0
        movsd xmm1, xmm1
        movsd xmm2, xmm2

        sub rsp, 8

loop:
        inc       i
        movsd     xmm0, [rsp]
        movsd     [rsp], xmm0
        movsd     xmm0, [rsp]
        movsd     [rsp], xmm0
        movsd     xmm0, [rsp]
        movsd     [rsp], xmm0
        cmp       i, N
        movsd     xmm0, [rsp]
        movsd     [rsp], xmm0
        movsd     xmm0, [rsp]
        movsd     [rsp], xmm0
        movsd     xmm0, [rsp]
        movsd     [rsp], xmm0
        jl        loop
done:

        add rsp, 8

        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
