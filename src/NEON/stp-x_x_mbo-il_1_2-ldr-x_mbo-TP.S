#define INSTR stp
#define NINST 4
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
loop:
        subs      x4, x4, #1
        INSTR    x0, x1, [sp, #-64] 
        ldr    x2, [sp, #-128] 
        ldr    x24, [sp, #-192]  
        INSTR    x6, x7, [sp, #-256]  
        ldr    x8, [sp, #-128]  
        ldr    x10, [sp, #-192]  
        INSTR    x12, x13, [sp, #-448]  
        ldr    x14, [sp, #-32]  
        ldr    x16, [sp, #-96]  
        INSTR    x18, x19, [sp, #-160]  
        ldr    x20, [sp, #-224]  
        ldr    x22, [sp, #-128]  
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
