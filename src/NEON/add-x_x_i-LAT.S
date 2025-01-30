#define INSTR add
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

        mov     x4, N

        fmov    v0.4s, #1.00000000
        fmov    v1.4s, #1.00000000
        mov     x0, #1
loop:
        subs      x4, x4, #1
        INSTR     x0, x0, #2 
        INSTR     x0, x0, #2 
        INSTR     x0, x0, #2 
        INSTR     x0, x0, #2 
        INSTR     x0, x0, #2 
        INSTR     x0, x0, #2 
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
        add            sp, sp, #64
        ld1            {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
