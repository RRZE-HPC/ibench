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

        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000
        mov     x24, #-64
        mov     x25, #-128
        mov     x28, #-192
        mov     x27, #-256
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
        INSTR    x10,  [sp]
        mul      x1, x1, x1
        INSTR    x11,  [sp]  
        mul      x2, x2, x2
        INSTR    x12,  [sp]  
        mul      x3, x3, x3
        INSTR    x13, [sp]  
        mul      x9, x9, x9
        INSTR    x14, [sp]  
        mul      x5, x5, x5
        INSTR    x15, [sp]  
        mul      x6, x6, x6
        INSTR    x16, [sp]  
        mul      x7, x7, x7
        INSTR    x17, [sp]  
        mul      x8, x8, x8
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
