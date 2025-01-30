#define INSTR vmovsd
#define NINST 12
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
loop:
        inc       i
        INSTR     xmm0, [rip+PI]
        INSTR     xmm1, [rip+PI]
        INSTR     xmm2, [rip+PI]
        INSTR     xmm3, [rip+PI]
        INSTR     xmm4, [rip+PI]
        INSTR     xmm5, [rip+PI]
        cmp       i, N
        INSTR     xmm6, [rip+PI]
        INSTR     xmm7, [rip+PI]
        INSTR     xmm8, [rip+PI]
        INSTR     xmm9, [rip+PI]
        INSTR     xmm10, [rip+PI]
        INSTR     xmm11, [rip+PI]
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
