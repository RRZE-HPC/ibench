#define INSTR vbroadcastss
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
loop:
        inc       i
        INSTR     zmm0, xmm6
        INSTR     zmm1, xmm7
        INSTR     zmm2, xmm8
        cmp       i, N
        INSTR     zmm3, xmm9
        INSTR     zmm4, xmm10
        INSTR     zmm5, xmm11
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
