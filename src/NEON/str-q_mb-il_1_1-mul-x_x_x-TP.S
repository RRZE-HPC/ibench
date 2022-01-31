#define INSTR str
#define NINST 8
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
        sub     x11, sp, #128
        sub     x12, sp, #256
        sub     x13, sp, #384
        sub     x14, sp, #512
        sub     x15, sp, #640
        sub     x16, sp, #768
        sub     x17, sp, #896
        sub     x18, sp, #1024

loop:
        subs      x4, x4, #1
        INSTR    q0,  [x11]
        mul      x1, x1, x1
        INSTR    q1,  [x12]  
        mul      x2, x2, x2
        INSTR    q2,  [x13]  
        mul      x3, x3, x3
        INSTR    q3, [x14] 
        mul      x5, x5, x5
        INSTR    q4, [x15]  
        mul      x6, x6, x6
        INSTR    q5, [x16]  
        mul      x7, x7, x7
        INSTR    q6, [x17]  
        mul      x8, x8, x8
        INSTR    q7, [x18]  
        mul      x9, x9, x9
        bne       loop
done:

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
