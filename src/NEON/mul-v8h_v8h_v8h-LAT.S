#define INSTR mul
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

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
loop:
        subs      x4, x4, #1
        INSTR     v0.8h, v0.8h, v1.8h
        INSTR     v0.8h, v0.8h, v1.8h
        INSTR     v0.8h, v0.8h, v1.8h
        INSTR     v0.8h, v0.8h, v1.8h
        INSTR     v0.8h, v0.8h, v1.8h
        INSTR     v0.8h, v0.8h, v1.8h
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
