#define INSTR vfmadd213sh
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
        INSTR     xmm10, xmm0, [rip+PI]
        INSTR     xmm11, xmm1, [rip+PI]
        INSTR     xmm12, xmm2, [rip+PI]
        cmp       i, N
        INSTR     xmm13, xmm3, [rip+PI]
        INSTR     xmm14, xmm4, [rip+PI]
        INSTR     xmm15, xmm5, [rip+PI]
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
