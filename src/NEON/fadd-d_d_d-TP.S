#define INSTR fadd
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

        fmov    d0, #1.00000000
        fmov    d1, #1.00000000
        fmov    d2, #1.00000000
loop:
        subs      x4, x4, #1
        INSTR     d3, d0, d0
        INSTR     d4, d0, d1
        INSTR     d5, d0, d2
        INSTR     d6, d1, d1
        INSTR     d7, d1, d2
        INSTR     d8, d2, d2
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
