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

        sub     x10, sp, #16
        sub     x11, sp, #32
        sub     x12, sp, #48
        sub     x13, sp, #64
        sub     x14, sp, #80
        sub     x15, sp, #96
        sub     x16, sp, #112
        sub     x17, sp, #128

loop:
        subs      x4, x4, #1
        INSTR    x10, [x10]
        mul      x1, x1, x1
        add      x2, x2, x2
        INSTR    x11, [x11]
        mul      x3, x3, x3
        add      x5, x5, x5
        INSTR    x12, [x12]
        mul      x6, x6, x6
        add      x7, x7, x7
        INSTR    x13, [x13]
        mul      x8, x8, x8
        add      x9, x9, x9
        INSTR    x14, [x14]
        mul      x18, x18, x18
        add      x19, x19, x19
        INSTR    x15, [x15]
        mul      x20, x20, x20
        add      x21, x21, x21
        INSTR    x16, [x16]
        mul      x22, x22, x22
        add      x24, x24, x24
        INSTR    x17, [x17]
        mul      x25, x25, x25
        add      x26, x26, x26
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
