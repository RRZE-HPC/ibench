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

        push      r9
        push      r10
        push      r11
        push      r12
        push      r13
        push      r14
        push      r15
loop:
        inc       i
        INSTR     [rip+PI], xmm0
        mov       [rip+PI], r9
        INSTR     [rip+PI], xmm1
        mov       [rip+PI], r10
        INSTR     [rip+PI], xmm2
        mov       [rip+PI], r12
        cmp       i, N
        INSTR     [rip+PI], xmm3
        mov       [rip+PI], r13 
        INSTR     [rip+PI], xmm4
        mov       [rip+PI], r14
        INSTR     [rip+PI], xmm5
        mov       [rip+PI], rcx 
        jl        loop
done:
        pop       r15
        pop       r14
        pop       r13
        pop       r12
        pop       r11
        pop       r10
        pop       r9
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
