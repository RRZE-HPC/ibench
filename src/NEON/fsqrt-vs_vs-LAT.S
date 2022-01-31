#define INSTR fsqrt
#define NINST 6
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

        fmov    v0.4s, #1.00000
        # create 2.0
        fadd v1.4s, v0.4s, v0.4s
        # create 3.0
        fadd v2.4s, v0.4s, v1.4s
        # create 4.0
        fadd v4.4s, v1.4s, v1.4s
        # create 8.0
        fadd v4.4s, v4.4s, v4.4s
        # create 16.0
        fadd v4.4s, v4.4s, v4.4s
        # create 32.0
        fadd v4.4s, v4.4s, v4.4s
        # create 64.0
        fadd v4.4s, v4.4s, v4.4s
        # create 128.0
        fadd v4.4s, v4.4s, v4.4s
        # create 256.0
        fadd v4.4s, v4.4s, v4.4s
        # create 512.0
        fadd v4.4s, v4.4s, v4.4s
        # create 1024.0
        fadd v4.4s, v4.4s, v4.4s
        # create 341.3333 = (1024.0/3.0)
        fdiv v1.4s, v4.4s, v2.4s
        # create 1/341.3333
        fdiv v2.4s, v0.4s, v1.4s
        # create 2*341.3333
        fadd v0.4s, v1.4s, v1.4s
        fadd v10.4s, v1.4s, v1.4s
loop:
        subs      x4, x4, #1
        INSTR     v1.4s, v0.4s
        # fadd      v1.4s, v1.4s, v10.4s
        INSTR     v2.4s, v1.4s
        # fadd      v2.4s, v2.4s, v10.4s
        INSTR     v3.4s, v2.4s
        # fadd      v3.4s, v3.4s, v10.4s
        INSTR     v4.4s, v3.4s
        # fadd      v4.4s, v4.4s, v10.4s
        INSTR     v5.4s, v4.4s
        # fadd      v5.4s, v5.4s, v10.4s
        INSTR     v0.4s, v5.4s
        # fadd      v0.4s, v0.4s, v10.4s
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
