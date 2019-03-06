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

        mov     x4, N

        fmov    v0.4s, #1.00000000
        fmov    v1.4s, #1.00000000
        fmov    v2.4s, #1.00000000
        fmov    v3.4s, #1.00000000

        fmov    v4.4s, #1.00000000
        fmov    v5.4s, #1.00000000
        fmov    v6.4s, #1.00000000
        fmov    v7.4s, #1.00000000

        fmov    v8.4s, #1.00000000
        fmov    v9.4s, #1.00000000
        fmov    v10.4s, #1.00000000
        fmov    v11.4s, #1.00000000

        fmov    v12.4s, #1.00000000
        fmov    v13.4s, #1.00000000
        fmov    v14.4s, #1.00000000
        fmov    v15.4s, #1.00000000

loop:
        subs      x4, x4, #1
        INSTR     v4.4s, v0.4s, v0.4s
        INSTR     v5.4s, v0.4s, v1.4s
        INSTR     v6.4s, v0.4s, v2.4s
        INSTR     v7.4s, v0.4s, v3.4s
        INSTR     v8.4s, v1.4s, v1.4s
        INSTR     v9.4s, v1.4s, v2.4s
        INSTR     v10.4s, v1.4s, v3.4s
        INSTR     v11.4s, v2.4s, v2.4s
        INSTR     v12.4s, v2.4s, v3.4s
        INSTR     v13.4s, v3.4s, v3.4s
        INSTR     v14.4s, v3.4s, v0.4s
        INSTR     v15.4s, v3.4s, v1.4s
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
        add            sp, sp, #64
        ld1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
