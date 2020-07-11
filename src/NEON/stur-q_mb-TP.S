#define INSTR stur
#define NINST 16
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
        mov     x24, #-64
        mov     x25, #-128
        mov     x28, #-192
        mov     x27, #-256

loop:
        subs      x4, x4, #1
        INSTR    q1,  [sp] 
        INSTR    q2,  [sp]  
        INSTR    q3,  [sp]  
        INSTR    q4, [sp]  
        INSTR    q5, [sp]  
        INSTR    q6, [sp]  
        INSTR    q7, [sp]  
        INSTR    q8, [sp]  
        INSTR    q9,  [sp] 
        INSTR    q10,  [sp]  
        INSTR    q11,  [sp]  
        INSTR    q12, [sp]  
        INSTR    q13, [sp]  
        INSTR    q14, [sp]  
        INSTR    q16, [sp]  
        INSTR    q18, [sp]  
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
