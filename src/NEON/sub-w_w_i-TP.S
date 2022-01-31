#define INSTR sub 
#define NINST 48
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
        mov     x1, #1
        mov     x2, #1
        mov     x3, #1
loop:
        subs      x4, x4, #1
        INSTR    w5, w1, #64
        INSTR    w6, w2, #128
        INSTR    w7, w3, #192
        INSTR    w8, w1, #256
        INSTR    w9, w2, #320
        INSTR    w10, w3, #384
        INSTR    w11, w1, #448
        INSTR    w12, w2, #512
        INSTR    w13, w3, #576
        INSTR    w14, w1, #640
        INSTR    w15, w2, #704
        INSTR    w16, w3, #764
        INSTR    w5, w1, #64
        INSTR    w6, w2, #128
        INSTR    w7, w3, #192
        INSTR    w8, w1, #256
        INSTR    w9, w2, #320
        INSTR    w10, w3, #384
        INSTR    w11, w1, #448
        INSTR    w12, w2, #512
        INSTR    w13, w3, #576
        INSTR    w14, w1, #640
        INSTR    w15, w2, #704
        INSTR    w16, w3, #764
        INSTR    w5, w1, #64
        INSTR    w6, w2, #128
        INSTR    w7, w3, #192
        INSTR    w8, w1, #256
        INSTR    w9, w2, #320
        INSTR    w10, w3, #384
        INSTR    w11, w1, #448
        INSTR    w12, w2, #512
        INSTR    w13, w3, #576
        INSTR    w14, w1, #640
        INSTR    w15, w2, #704
        INSTR    w16, w3, #764
        INSTR    w5, w1, #64
        INSTR    w6, w2, #128
        INSTR    w7, w3, #192
        INSTR    w8, w1, #256
        INSTR    w9, w2, #320
        INSTR    w10, w3, #384
        INSTR    w11, w1, #448
        INSTR    w12, w2, #512
        INSTR    w13, w3, #576
        INSTR    w14, w1, #640
        INSTR    w15, w2, #704
        INSTR    w16, w3, #764
        bne       loop
done:

        # pop callee-save registers from stack
	    ldp x19, x20, [sp, 16]
	    ldp x21, x22, [sp, 32]
	    ldp x24, x25, [sp, 48]
	    ldp x26, x27, [sp, 64]
	    ldr x28, [sp, 80]
	    ldp x29, x30, [sp], 96
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
