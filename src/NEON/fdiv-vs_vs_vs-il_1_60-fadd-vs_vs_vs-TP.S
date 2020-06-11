#define INSTR fdiv
#define NINST 1
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
loop:
        subs      x4, x4, #1
        INSTR     v3.4s, v0.4s, v1.4s
        fadd      v4.4s, v0.4s, v0.4s
        fadd      v5.4s, v0.4s, v1.4s
        fadd      v6.4s, v0.4s, v2.4s
        fadd      v7.4s, v0.4s, v0.4s
        fadd      v8.4s, v0.4s, v1.4s
        fadd      v9.4s, v0.4s, v2.4s
        fadd      v10.4s, v0.4s, v0.4s
        fadd      v11.4s, v0.4s, v1.4s
        fadd      v12.4s, v0.4s, v2.4s
        fadd      v13.4s, v0.4s, v0.4s
        fadd      v14.4s, v0.4s, v1.4s
        fadd      v15.4s, v0.4s, v2.4s
        fadd      v16.4s, v0.4s, v0.4s
        fadd      v17.4s, v0.4s, v1.4s
        fadd      v18.4s, v0.4s, v2.4s
        fadd      v4.4s, v0.4s, v0.4s
        fadd      v5.4s, v0.4s, v1.4s
        fadd      v6.4s, v0.4s, v2.4s
        fadd      v7.4s, v0.4s, v0.4s
        fadd      v8.4s, v0.4s, v1.4s
        fadd      v9.4s, v0.4s, v2.4s
        fadd      v10.4s, v0.4s, v0.4s
        fadd      v11.4s, v0.4s, v1.4s
        fadd      v12.4s, v0.4s, v2.4s
        fadd      v13.4s, v0.4s, v0.4s
        fadd      v14.4s, v0.4s, v1.4s
        fadd      v15.4s, v0.4s, v2.4s
        fadd      v16.4s, v0.4s, v0.4s
        fadd      v17.4s, v0.4s, v1.4s
        fadd      v18.4s, v0.4s, v2.4s
        fadd      v4.4s, v0.4s, v0.4s
        fadd      v5.4s, v0.4s, v1.4s
        fadd      v6.4s, v0.4s, v2.4s
        fadd      v7.4s, v0.4s, v0.4s
        fadd      v8.4s, v0.4s, v1.4s
        fadd      v9.4s, v0.4s, v2.4s
        fadd      v10.4s, v0.4s, v0.4s
        fadd      v11.4s, v0.4s, v1.4s
        fadd      v12.4s, v0.4s, v2.4s
        fadd      v13.4s, v0.4s, v0.4s
        fadd      v14.4s, v0.4s, v1.4s
        fadd      v15.4s, v0.4s, v2.4s
        fadd      v16.4s, v0.4s, v0.4s
        fadd      v17.4s, v0.4s, v1.4s
        fadd      v18.4s, v0.4s, v2.4s
        fadd      v4.4s, v0.4s, v0.4s
        fadd      v5.4s, v0.4s, v1.4s
        fadd      v6.4s, v0.4s, v2.4s
        fadd      v7.4s, v0.4s, v0.4s
        fadd      v8.4s, v0.4s, v1.4s
        fadd      v9.4s, v0.4s, v2.4s
        fadd      v10.4s, v0.4s, v0.4s
        fadd      v11.4s, v0.4s, v1.4s
        fadd      v12.4s, v0.4s, v2.4s
        fadd      v13.4s, v0.4s, v0.4s
        fadd      v14.4s, v0.4s, v1.4s
        fadd      v15.4s, v0.4s, v2.4s
        fadd      v16.4s, v0.4s, v0.4s
        fadd      v17.4s, v0.4s, v1.4s
        fadd      v18.4s, v0.4s, v2.4s
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
        add            sp, sp, #64
        ld1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
