#define INSTR ldr
#define NINST 8
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
	str      x2,  [x24]	 
        INSTR    x3,  [sp] 
	str      x5,  [x24]	 
        INSTR    x6,  [sp] 
	str      x7,  [x24]	 
        INSTR    x8,  [sp] 
	str      x9,  [x24]
        INSTR    x10,  [sp]  
	str      x11,  [x24]
        INSTR    x12,  [sp]  
	str      x13,  [x24]
        INSTR    x14, [sp]  
	str      x15,  [x24]
        INSTR    x16, [sp]  
	str      x17,  [x24]
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
