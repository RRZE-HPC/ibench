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
        # sub            sp, sp, #64
        # st1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        # sub            sp, sp, #64
        # st1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
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
        mov     x24, sp

        mov     x16, sp
        add     x16, x16, #128
        mov     x25, sp
        add     x25, x25, #192
        mov     x27, sp
        sub     x27, x27, #256
        mov     x28, sp
        sub     x28, x28, #320
        mov     x1, #1
        mov     x2, #1
        mov     x3, #1
        mov     x5, #1
        mov     x6, #1
        mov     x7, #1
        mov     x8, #1
        mov     x9, #1
        mov     x10, #1
        mov     x11, #1
        mov     x12, #1
        mov     x13, #1
        mov     x14, #1
        mov     x15, #1
        mov     x17, #1
        mov     x18, #1
loop:
        subs      x4, x4, #1
        INSTR    q1, [x25], #64
        mul      x1, x1, x1
        add      x2, x2, x2
        INSTR    q2, [x27], #64
        mul      x3, x3, x3
        add      x5, x5, x5
        INSTR    q3, [x28], #64
        mul      x6, x6, x6
        add      x7, x7, x7
        INSTR    q4, [x16], #64
        mul      x8, x8, x8
        add      x9, x9, x9
        INSTR    q6, [x25], #-64
        mul      x10, x10, x10
        add      x11, x11, x11
        INSTR    q7, [x27], #-64 
        mul      x12, x8, x12
        add      x13, x13, x13
        INSTR    q8, [x28], #-64
        mul      x14, x14, x14
        add      x15, x15, x15
        INSTR    q9, [x16], #-64
        mul      x17, x17, x17
        add      x18, x18, x18
        bne       loop
done:
        mov     sp, x24
        # pop callee-save registers from stack
        ldp x19, x20, [sp, 16]
        ldp x21, x22, [sp, 32]
        ldp x24, x25, [sp, 48]
        ldp x26, x27, [sp, 64]
        ldr x28, [sp, 80]
        ldp x29, x30, [sp], 96
        # ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        # add            sp, sp, #64
        # ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        # add            sp, sp, #64
        
        ret

.size latency, .-latency
