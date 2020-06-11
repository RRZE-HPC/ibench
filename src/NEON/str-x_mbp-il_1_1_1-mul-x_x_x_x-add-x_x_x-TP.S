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

        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000
        mov     x24, sp

        mov     x14, sp
        add     x14, x14, #128
        mov     x15, sp
        add     x15, x15, #192
        mov     x16, sp
        sub     x16, x16, #256
        mov     x17, sp
        sub     x17, x17, #320
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
        mov     x12, #1
        mov     x13, #1
        mov     x18, #1
        mov     x25, #1
        mov     x27, #1
        mov     x28, #1

loop:
        subs      x4, x4, #1
        INSTR    x24, [x14], #64
        mul      x1, x1, x1
        add      x2, x2, x2
        INSTR    x24, [x15], #64
        mul      x3, x3, x3
        add      x5, x5, x5
        INSTR    x24, [x16], #64
        mul      x6, x6, x6
        add      x7, x7, x7
        INSTR    x24, [x14], #-64
        mul      x8, x8, x8
        add      x9, x9, x9
        INSTR    x24, [x15], #-64
        mul      x10, x10, x10
        add      x11, x11, x11
        INSTR    x24, [x16], #-64 
        mul      x12, x12, x12
        add      x13, x13, x13
        INSTR    x24, [x17], #64
        mul      x18, x18, x18
        add      x25, x25, x25
        INSTR    x24, [x17], #-64
        mul      x27, x27, x27
        add      x28, x28, x28
        bne       loop
done:
        mov     sp, x24
        # pop callee-save registers from stack
        # ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        # add            sp, sp, #64
        # ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        # add            sp, sp, #64
        
        ret

.size latency, .-latency
