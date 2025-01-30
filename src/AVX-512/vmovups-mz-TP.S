#define INSTR vmovups
#define NINST 6
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
        INSTR     [rip+PI], zmm0
        INSTR     [rip+PI], zmm1
        INSTR     [rip+PI], zmm2
        cmp       i, N
        INSTR     [rip+PI], zmm3
        INSTR     [rip+PI], zmm4
        INSTR     [rip+PI], zmm5
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
