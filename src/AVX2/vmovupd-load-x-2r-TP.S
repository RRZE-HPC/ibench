#define INSTR vmovupd
#define NINST 5
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
        INSTR     xmm0, [rip+PI]
        mov       rax, [rip+PI] 
        mov       rbx, [rip+PI] 
        INSTR     xmm1, [rip+PI]
        mov       rcx, [rip+PI] 
        mov       rdx, [rip+PI] 
        INSTR     xmm2, [rip+PI]
        mov       r9, [rip+PI] 
        mov       r10, [rip+PI] 
        cmp       i, N
        INSTR     xmm3, [rip+PI]
        mov       r11, [rip+PI] 
        mov       r12, [rip+PI] 
        INSTR     xmm4, [rip+PI]
        mov       r13, [rip+PI] 
        mov       r14, [rip+PI] 
        //INSTR     xmm5, [rip+PI]
        //mov       rcx, [rip+PI] 
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
