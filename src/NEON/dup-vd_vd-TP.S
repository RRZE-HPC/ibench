#define INSTR dup
#define NINST 13
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
loop:
        subs      x4, x4, #1
        INSTR     v0.2d, v8.d[0]
        INSTR     v1.2d, v9.d[0]
        INSTR     v2.2d, v10.d[0]
        INSTR     v3.2d, v11.d[0]
        INSTR     v4.2d, v12.d[0]
        INSTR     v5.2d, v13.d[0]
        INSTR     v6.2d, v14.d[0]
        INSTR     v7.2d, v15.d[0]
        INSTR     v16.2d, v21.d[0]
        INSTR     v17.2d, v22.d[0]
        INSTR     v18.2d, v23.d[0]
        INSTR     v19.2d, v24.d[0]
        INSTR     v20.2d, v25.d[0]
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
