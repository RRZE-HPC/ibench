#define INSTR ldr
#define NINST 18
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
        sub     sp, sp, #64
        st1     {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        sub     sp, sp, #64
        st1     {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        sub     sp, sp, #64
        st1     {v16.2d, v17.2d, v18.2d, v19.2d}, [sp]
        sub     sp, sp, #64
        st1     {v20.2d, v21.2d, v22.2d, v23.2d}, [sp]
        sub     sp, sp, #64
        st1     {v24.2d, v25.2d, v26.2d, v27.2d}, [sp]
        sub     sp, sp, #64
        st1     {v28.2d, v29.2d, v30.2d, v31.2d}, [sp]
        stp     x19, x20, [sp, -96]!
        stp     x21, x22, [sp, 16]
        stp     x23, x24, [sp, 32]
        stp     x25, x26, [sp, 48]
        stp     x27, x28, [sp, 64]
        stp     x29, x30, [sp, 80]

        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000

        add     x11, sp, #32
        add     x12, sp, #64
        add     x13, sp, #96
        add     x14, sp, #128
        add     x15, sp, #160
        add     x16, sp, #192
        add     x17, sp, #224
        add     x18, sp, #256

loop:
        subs      x4, x4, #1
        INSTR    q0, [sp],  #16
        INSTR    q1, [x11], #16
        INSTR    q2, [x12], #16
        INSTR    q3, [x13], #16
        INSTR    q4, [x14], #16
        INSTR    q5, [x15], #16
        INSTR    q6, [x16], #16
        INSTR    q7, [x17], #16
        INSTR    q8, [x18], #16
        INSTR    q9, [sp],  #-16 
        INSTR    q10, [x11], #-16
        INSTR    q11, [x12], #-16 
        INSTR    q12, [x13], #-16
        INSTR    q13, [x14], #-16
        INSTR    q14, [x15], #-16
        INSTR    q15, [x16], #-16
        INSTR    q16, [x17], #-16
        INSTR    q17, [x18], #-16
        bne       loop
done:
        # pop callee-save registers from stack
        ldp     x19, x20, [sp]
        ldp     x21, x22, [sp, 16]
        ldp     x23, x24, [sp, 32]
        ldp     x25, x26, [sp, 48]
        ldp     x27, x28, [sp, 64]
        ldp     x29, x30, [sp, 80]
        add     sp, sp, #96
        ld1     {v28.2d, v29.2d, v30.2d, v31.2d}, [sp], #64
        ld1     {v24.2d, v25.2d, v26.2d, v27.2d}, [sp], #64
        ld1     {v20.2d, v21.2d, v22.2d, v23.2d}, [sp], #64
        ld1     {v16.2d, v17.2d, v18.2d, v19.2d}, [sp], #64
        ld1     {v12.2d, v13.2d, v14.2d, v15.2d}, [sp], #64
        ld1     {v8.2d, v9.2d, v10.2d, v11.2d}, [sp], #64
        
        ret

.size latency, .-latency
