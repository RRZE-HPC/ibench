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

        add     x10, sp, #-64
        sub     x11, x10, #448
        sub     x12, x10, #320
        sub     x13, x10, #192
        sub     x14, x10, #64

loop:
        subs      x4, x4, #1
        INSTR    x10, [x11], #64
        mul      x1, x1, x1
        add      x2, x2, x2
        INSTR    x10, [x12], #64
        mul      x3, x3, x3
        add      x5, x5, x5
        INSTR    x10, [x13], #64
        mul      x6, x6, x6
        add      x7, x7, x7
        INSTR    x10, [x14], #64
        mul      x8, x8, x8
        add      x9, x9, x9
        INSTR    x10, [x11], #-64
        mul      x15, x15, x15
        add      x16, x16, x16
        INSTR    x10, [x12], #-64 
        mul      x17, x17, x17
        add      x18, x18, x18
        INSTR    x10, [x13], #-64
        mul      x19, x19, x19
        add      x20, x20, x20
        INSTR    x10, [x14], #-64
        mul      x21, x21, x21
        add      x22, x22, x22
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
