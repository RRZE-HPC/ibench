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
        mov     x10, sp

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

loop:
        subs      x4, x4, #1
        INSTR    x11, [x25], #64
        mul      x1, x1, x1
        INSTR    x12, [x27], #64
        mul      x2, x2, x2
        INSTR    x13, [x28], #64
        mul      x3, x3, x3
        INSTR    x14, [x16], #64
        mul      x9, x9, x9
        INSTR    x15, [x25], #-64
        mul      x5, x5, x5
        INSTR    x17, [x27], #-64 
        mul      x6, x6, x6
        INSTR    x18, [x28], #-64
        mul      x7, x7, x7
        INSTR    x19, [x16], #-64
        mul      x8, x8, x8
        bne       loop
done:
        mov     sp, x10
        # pop callee-save registers from stack
        # ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        # add            sp, sp, #64
        # ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        # add            sp, sp, #64
        
        ret

.size latency, .-latency
