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
        sub            sp, sp, #64
        st1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        sub            sp, sp, #64
        st1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        stp x29, x30, [sp, -96]!
        stp x19, x20, [sp, 16]
        stp x21, x22, [sp, 32]
        stp x24, x25, [sp, 48]
        stp x26, x27, [sp, 64]
        str x28, [sp, 80]

        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000

        sub     sp, sp, #-128
        mov     x24, sp
        mov     x10, sp
        mov     x11, sp
        mov     x12, sp
        mov     x13, sp
        add     x10, x10, #16
        add     x11, x11, #48
        sub     x12, x12, #16
        sub     x13, x13, #48
        mov     x16, sp
        add     x16, x16, #32
        mov     x25, sp
        add     x25, x25, #64
        mov     x27, sp
        sub     x27, x27, #32
        mov     x28, sp
        sub     x28, x28, #64
loop:
        subs      x4, x4, #1
        INSTR    q0, [sp],  #64
        INSTR    q1, [x25], #64
        INSTR    q2, [x27], #64
        INSTR    q3, [x28], #64
        INSTR    q4, [x16], #64
        INSTR    q5, [x10], #64
        INSTR    q6, [x11], #64
        INSTR    q6, [x12], #64
        INSTR    q6, [x13], #64
        INSTR    q7, [sp],  #-64 
        INSTR    q8, [x25], #-64
        INSTR    q9, [x27], #-64 
        INSTR    q10, [x28], #-64
        INSTR    q11, [x16], #-64
        INSTR    q12, [x10], #-64
        INSTR    q13, [x11], #-64
        INSTR    q6, [x12], #-64
        INSTR    q6, [x13], #-64
        bne       loop
done:
        mov     sp, x24
        add     sp, sp, #-128
        # pop callee-save registers from stack
        ldp x19, x20, [sp, 16]
        ldp x21, x22, [sp, 32]
        ldp x24, x25, [sp, 48]
        ldp x26, x27, [sp, 64]
        ldr x28, [sp, 80]
        ldp x29, x30, [sp], 96
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
