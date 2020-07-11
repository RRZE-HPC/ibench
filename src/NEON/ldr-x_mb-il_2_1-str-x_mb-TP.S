#define INSTR ldr
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
        mov     x24, sp
	sub	x24, x24, #192

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000

loop:
        subs      x4, x4, #1
        INSTR    x1,  [sp] 
        INSTR    x2,  [sp] 
	str      x3,  [x24]	 
        INSTR    x5,  [sp] 
        INSTR    x6,  [sp] 
	str      x7,  [x24]
        INSTR    x8,  [sp]  
        INSTR    x9,  [sp]  
	str      x10,  [x24]
        INSTR    x11,  [sp]  
        INSTR    x12,  [sp]  
	str      x13,  [x24]
        INSTR    x14, [sp]  
        INSTR    x15,  [sp]  
	str      x16,  [x24]
        INSTR    x17, [sp]  
        INSTR    x18,  [sp]  
	str      x28,  [x24]
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
