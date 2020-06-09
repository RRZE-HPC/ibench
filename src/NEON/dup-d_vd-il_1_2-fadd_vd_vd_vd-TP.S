#define INSTR dup
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
        fmov    v2.2d, #1.00000000
        fmov    v12.2d, #1.00000000
loop:
        subs      x4, x4, #1
        
        #add       x5, x0, x0
        #add       x6, x0, x0
        #add       x7, x0, x0
        #add       x8, x0, x0
        #add       x9, x0, x0
        #add       x10, x0, x0
        
        fadd      v13.2d, v12.2d, v12.2d
        fadd      v14.2d, v12.2d, v12.2d
        INSTR     d0, v6.d[0]
        fadd      v15.2d, v12.2d, v12.2d
        fadd      v16.2d, v12.2d, v12.2d
        INSTR     d1, v7.d[1]
        fadd      v17.2d, v12.2d, v12.2d
        fadd      v18.2d, v12.2d, v12.2d
        INSTR     d2, v8.d[0]

        fadd      v19.2d, v12.2d, v12.2d
        fadd      v20.2d, v12.2d, v12.2d
        INSTR     d3, v9.d[1]
        fadd      v21.2d, v12.2d, v12.2d
        fadd      v22.2d, v12.2d, v12.2d
        INSTR     d4, v10.d[0]
        fadd      v23.2d, v12.2d, v12.2d
        fadd      v24.2d, v12.2d, v12.2d
        INSTR     d5, v11.d[1]
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
