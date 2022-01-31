#define INSTR fmadd
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

        fmov    d0, #1.00000000
        fmov    d1, #1.00000000
        fmov    d2, #1.00000000
loop:
        subs      x4, x4, #1
        INSTR     s3, s0, s0, s0
        INSTR     s4, s0, s1, s1
        INSTR     s5, s0, s2, s2
        INSTR     s6, s1, s1, s1
        INSTR     s7, s1, s2, s2
        INSTR     s8, s2, s2, s2
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
