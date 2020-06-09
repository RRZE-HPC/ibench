#define INSTR cmp
#define NINST 192
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
        mov     x1, #2
        mov     x2, #3
        mov     x3, #4
loop:
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3

        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3

        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3

        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1 
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        INSTR    w5, w1
        INSTR    w6, w2
        INSTR    w7, w3
        INSTR    w8, w1
        INSTR    w9, w2
        INSTR    w10, w3
        INSTR    w11, w1
        INSTR    w12, w2
        INSTR    w13, w3
        INSTR    w14, w1
        INSTR    w15, w2
        INSTR    w16, w3
        subs      x4, x4, #1
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
