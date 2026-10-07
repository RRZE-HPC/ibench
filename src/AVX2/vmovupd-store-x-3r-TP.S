#define INSTR vmovupd
#define NINST 3
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

        push      rax
        push      rbx
        push      rcx
        push      rdx
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
        mov       [rip+PI], rax 
        mov       [rip+PI], rbx 
        mov       [rip+PI], rcx 
        INSTR     [rip+PI], xmm1
        mov       [rip+PI], rdx 
        mov       [rip+PI], r9 
        mov       [rip+PI], r10 
        cmp       i, N
        INSTR     [rip+PI], xmm2
        mov       [rip+PI], r11 
        mov       [rip+PI], r12 
        mov       [rip+PI], r13 
        jl        loop
done:
        pop       r15
        pop       r14
        pop       r13
        pop       r12
        pop       r11
        pop       r10
        pop       r9
        pop       rdx
        pop       rcx
        pop       rbx
        pop       rax
        mov  rsp, rbp
        pop rbp
        ret
.size latency, .-latency
