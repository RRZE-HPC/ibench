#define INSTR vmovupd
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
        INSTR     ymm0, [rip+PI]
        INSTR     ymm1, [rip+PI]
        INSTR     ymm2, [rip+PI]
        cmp       i, N
        INSTR     ymm3, [rip+PI]
        INSTR     ymm4, [rip+PI]
        INSTR     ymm5, [rip+PI]
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
