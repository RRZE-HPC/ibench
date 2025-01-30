#define INSTR ldr
#define NINST 12
#define N x0

.globl ninst
.data
ninst:
.long NINST
.text
.globl latency
.type latency, @function
.align 2
latency:

        # push callee-save registers onto stack
        sub            sp, sp, #64
        st1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        sub            sp, sp, #64
        st1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]

        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000
        mov     x24, sp
loop:
        subs      x4, x4, #1
        INSTR    q0, [sp], #64
        INSTR    q1, [sp], #64
        INSTR    q2, [sp], #64
        INSTR    q3, [sp], #64
        INSTR    q4, [sp], #64 
        INSTR    q5, [sp], #64
        INSTR    q6, [sp], #-64 
        INSTR    q7, [sp], #-64
        INSTR    q8, [sp], #-64 
        INSTR    q9, [sp], #-64 
        INSTR    q10, [sp], #-64 
        INSTR    q11, [sp], #-64

        bne       loop
done:
        mov     sp, x24
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
