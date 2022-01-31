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
        
        sub     sp, sp, #256
        sub     x1, sp, #896
        sub     x2, sp, #640
        sub     x3, sp, #384
        sub     x5, sp, #128

loop:
        subs     x4, x4, #1
        INSTR    q1, [x1], #128
        mul      x11, x11, x11
        add      x12, x12, x12
        INSTR    q2, [x2], #128
        mul      x13, x13, x13
        add      x14, x14, x14
        INSTR    q3, [x3], #128
        mul      x15, x15, x15
        add      x16, x16, x16
        INSTR    q4, [x5], #128
        mul      x17, x17, x17
        add      x18, x18, x18
        INSTR    q6, [x1], #-128
        mul      x19, x19, x19
        add      x20, x20, x20
        INSTR    q7, [x2], #-128
        mul      x21, x21, x21
        add      x22, x22, x22
        INSTR    q8, [x3], #-128
        mul      x24, x24, x24
        add      x25, x25, x25
        INSTR    q9, [x5], #-128
        mul      x26, x26, x26
        add      x27, x27, x27
        bne       loop
done:
        add     sp, sp, #256
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
