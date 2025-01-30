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
        INSTR     xmm10, xmm0, xmm20 
        INSTR     xmm11, xmm1, xmm21
        INSTR     xmm12, xmm2, xmm22
        cmp       i, N
        INSTR     xmm13, xmm3, xmm23
        INSTR     xmm14, xmm4, xmm24
        INSTR     xmm15, xmm5, xmm25
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
