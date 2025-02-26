#define INSTR vpbroadcastd
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
        INSTR     ymm6,  r9d
        INSTR     ymm7,  r10d
        INSTR     ymm8,  r11d
        cmp       i, N
        INSTR     ymm9,  r12d
        INSTR     ymm10, r13d
        INSTR     ymm11, r14d
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
