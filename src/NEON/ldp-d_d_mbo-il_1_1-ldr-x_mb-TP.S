#define INSTR ldp
#define NINST 12
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
loop:
        subs      x4, x4, #1
        INSTR    d24, d1, [sp, #-256]
        ldr      x0, [sp]
        INSTR    d2,  d3, [sp, #-256] 
        ldr      x1, [sp]
        INSTR    d25, d5, [sp, #-256]  
        ldr      x2, [sp]
        INSTR    d6,  d7, [sp, #-256]  
        ldr      x3, [sp]
        INSTR    d8,  d9, [sp, #-256]  
        ldr      x5, [sp]
        INSTR    d10, d11, [sp, #-256]  
        ldr      x6, [sp]
        INSTR    d12, d13, [sp, #-256]  
        ldr      x7, [sp]
        INSTR    d14, d15, [sp, #-256]  
        ldr      x8, [sp]
        INSTR    d16, d17, [sp, #-256]  
        ldr      x9, [sp]
        INSTR    d18, d19, [sp, #-256]  
        ldr      x10, [sp]
        INSTR    d20, d21, [sp, #-256]  
        ldr      x11, [sp]
        INSTR    d22, d23, [sp, #-256]  
        ldr      x12, [sp]
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
