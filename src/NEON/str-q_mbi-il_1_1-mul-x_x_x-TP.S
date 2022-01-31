#define INSTR str
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
        fmov    v3.2d, #1.00000000
        fmov    v4.2d, #1.00000000
        fmov    v5.2d, #1.00000000
        fmov    v6.2d, #1.00000000
        fmov    v7.2d, #1.00000000
        mov     x9, #-64
        mov     x10, #-128
        mov     x11, #-192
        mov     x12, #-256
        mov     x13, #-320
        mov     x14, #-384
        mov     x15, #-450
        mov     x16, #-512
        mov     x0, #1
        mov     x1, #1
        mov     x2, #1
        mov     x3, #1
        mov     x5, #1
        mov     x6, #1
        mov     x7, #1
        mov     x8, #1

loop:
        subs      x4, x4, #1
        INSTR    q0,  [sp, x9]
        mul      x0, x0, x0
        INSTR    q1,  [sp, x10]  
        mul      x1, x1, x1
        INSTR    q2,  [sp, x11]  
        mul      x2, x2, x2
        INSTR    q3, [sp, x12]  
        mul      x3, x3, x3
        INSTR    q4, [sp, x13]  
        mul      x5, x5, x5
        INSTR    q5, [sp, x14]  
        mul      x6, x6, x6
        # INSTR    q6, [sp, x15]  
        # mul      x7, x7, x7
        # INSTR    q7, [sp, x16]  
        # mul      x8, x8, x8
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
