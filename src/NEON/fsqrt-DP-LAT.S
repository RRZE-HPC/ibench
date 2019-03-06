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
        st1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        sub            sp, sp, #64
        st1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]

        mov     x4, N

        fmov    v0.2d, #1.00000
        # create 2.0
        fadd v1.2d, v0.2d, v0.2d
        # create 3.0
        fadd v2.2d, v0.2d, v1.2d
        # create 4.0
        fadd v4.2d, v1.2d, v1.2d
        # create 8.0
        fadd v4.2d, v4.2d, v4.2d
        # create 16.0
        fadd v4.2d, v4.2d, v4.2d
        # create 32.0
        fadd v4.2d, v4.2d, v4.2d
        # create 64.0
        fadd v4.2d, v4.2d, v4.2d
        # create 128.0
        fadd v4.2d, v4.2d, v4.2d
        # create 256.0
        fadd v4.2d, v4.2d, v4.2d
        # create 512.0
        fadd v4.2d, v4.2d, v4.2d
        # create 1024.0
        fadd v4.2d, v4.2d, v4.2d
        # create 341.3333 = (1024.0/3.0)
        fdiv v1.2d, v4.2d, v2.2d
        # create 1/341.3333
        fdiv v2.2d, v0.2d, v1.2d
        # create 2*341.3333
        fadd v0.2d, v1.2d, v1.2d
        fadd v10.2d, v1.2d, v1.2d
loop:
        subs      x4, x4, #1
        INSTR     v1.2d, v0.2d
        fadd      v1.2d, v1.2d, v10.2d
        INSTR     v2.2d, v1.2d
        fadd      v2.2d, v2.2d, v10.2d
        INSTR     v3.2d, v2.2d
        fadd      v3.2d, v3.2d, v10.2d
        INSTR     v4.2d, v3.2d
        fadd      v4.2d, v4.2d, v10.2d
        INSTR     v5.2d, v4.2d
        fadd      v5.2d, v5.2d, v10.2d
        INSTR     v0.2d, v5.2d
        fadd      v0.2d, v0.2d, v10.2d
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
