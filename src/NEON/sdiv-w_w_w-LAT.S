#define INSTR sdiv
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
        st1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        sub            sp, sp, #64
        st1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
	stp x29, x30, [sp, -96]!
	stp x19, x20, [sp, 16]
	stp x21, x22, [sp, 32]
	stp x24, x25, [sp, 48]
	stp x26, x27, [sp, 64]
	str x28, [sp, 80]

        mov     x4, N

        fmov    v0.4s, #1.00000000
        fmov    v1.4s, #1.00000000
        mov     x0, #1
        mov     x1, #1
loop:
        subs      x4, x4, #1
        INSTR     w0, w0, w1
        INSTR     w0,  w0, w1
        INSTR     w0,  w0, w1
        INSTR     w0,  w0, w1
        INSTR     w0,  w0, w1
        INSTR     w0,  w0, w1
        bne       loop
done:

        # pop callee-save registers from stack
	ldp x19, x20, [sp, 16]
	ldp x21, x22, [sp, 32]
	ldp x24, x25, [sp, 48]
	ldp x26, x27, [sp, 64]
	ldr x28, [sp, 80]
	ldp x29, x30, [sp], 96
        ld1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
        add            sp, sp, #64
        ld1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
