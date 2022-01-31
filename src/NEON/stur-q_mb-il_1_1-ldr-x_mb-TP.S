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

        mov      x30, sp
        sub     sp, sp, #128
        # remove the line below for dependency chain
        sub     x30, sp, #128
loop:
        subs      x4, x4, #1
        INSTR    q1,  [sp]
        ldr      x0, [x30]
        INSTR    q2,  [sp]  
        ldr      x1, [x30]
        INSTR    q3,  [sp]  
        ldr      x2, [x30]
        INSTR    q4, [sp]  
        ldr      x3, [x30]
        INSTR    q5, [sp]  
        ldr      x5, [x30]
        INSTR    q6, [sp]  
        ldr      x6, [x30]
        INSTR    q7, [sp]  
        ldr      x7, [x30]
        INSTR    q8, [sp]  
        ldr      x8, [x30]
        INSTR    q9,  [sp] 
        ldr      x9, [x30]
        INSTR    q10,  [sp]  
        ldr      x10, [x30]
        INSTR    q11,  [sp]  
        ldr      x11, [x30]
        INSTR    q12, [sp]  
        ldr      x12, [x30]
        INSTR    q13, [sp]  
        ldr      x13, [x30]
        INSTR    q14, [sp]  
        ldr      x14, [x30]
        INSTR    q16, [sp]  
        ldr      x15, [x30]
        INSTR    q18, [sp]  
        ldr      x16, [x30]
        bne       loop
done:
        add     sp, sp, #128
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
