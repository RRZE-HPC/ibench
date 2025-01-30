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
        INSTR     r9d,  ymm6
        INSTR     r10d, ymm7
        INSTR     r11d, ymm8
        cmp       i, N
        INSTR     r12d, ymm9
        INSTR     r13d, ymm10
        INSTR     r14d, ymm11
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
