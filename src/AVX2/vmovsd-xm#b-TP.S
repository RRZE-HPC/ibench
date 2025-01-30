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
        INSTR     [rip+PI], xmm0 
        INSTR     [rip+PI], xmm1 
        INSTR     [rip+PI], xmm2 
        INSTR     [rip+PI], xmm3 
        INSTR     [rip+PI], xmm4 
        INSTR     [rip+PI], xmm5 
        cmp       i, N
        INSTR     [rip+PI], xmm6 
        INSTR     [rip+PI], xmm7 
        INSTR     [rip+PI], xmm8 
        INSTR     [rip+PI], xmm9 
        INSTR     [rip+PI], xmm10
        INSTR     [rip+PI], xmm11
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
