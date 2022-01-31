#define INSTR fmla
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
        st1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        sub            sp, sp, #64
        st1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
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
        fmov    v3.2d, #1.00000000

        fmov    v4.2d, #1.00000000
        fmov    v5.2d, #1.00000000
        fmov    v6.2d, #1.00000000
        fmov    v7.2d, #1.00000000

        fmov    v8.2d, #1.00000000
        fmov    v9.2d, #1.00000000
        fmov    v10.2d, #1.00000000
        fmov    v11.2d, #1.00000000

        fmov    v12.2d, #1.00000000
        fmov    v13.2d, #1.00000000
        fmov    v14.2d, #1.00000000
        fmov    v15.2d, #1.00000000

loop:
        subs      x4, x4, #1
        INSTR     v4.2d, v0.2d, v0.2d
        INSTR     v5.2d, v0.2d, v1.2d
        INSTR     v6.2d, v0.2d, v2.2d
        INSTR     v7.2d, v0.2d, v3.2d
        INSTR     v8.2d, v1.2d, v1.2d
        INSTR     v9.2d, v1.2d, v2.2d
        INSTR     v10.2d, v1.2d, v3.2d
        INSTR     v11.2d, v2.2d, v2.2d
        INSTR     v12.2d, v2.2d, v3.2d
        INSTR     v13.2d, v3.2d, v3.2d
        INSTR     v14.2d, v3.2d, v0.2d
        INSTR     v15.2d, v3.2d, v1.2d
        bne       loop
done:

        # pop callee-save registers from stack
        ldp x19, x20, [sp, 16]
        ldp x21, x22, [sp, 32]
        ldp x24, x25, [sp, 48]
        ldp x26, x27, [sp, 64]
        ldr x28, [sp, 80]
        ldp x29, x30, [sp], 96
        ld1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
        add            sp, sp, #64
        ld1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
