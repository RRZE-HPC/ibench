#define INSTR vfmadd213ph
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
        INSTR     zmm10, zmm0, zmm20
        INSTR     zmm11, zmm1, zmm21
        INSTR     zmm12, zmm2, zmm22
        cmp       i, N
        INSTR     zmm13, zmm3, zmm23
        INSTR     zmm14, zmm4, zmm24
        INSTR     zmm15, zmm5, zmm25
        jl        loop
done:
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
