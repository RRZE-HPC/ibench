#define INSTR mov 
#define NINST 15
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

        mov    x0, #1
        mov    x1, #2
        mov    x2, #3
loop:
        subs      x4, x4, #1
        INSTR     x3, #3
        INSTR     x5, #5
        INSTR     x6, #6
        INSTR     x7, #7
        INSTR     x8, #8
        INSTR     x9, #9
        INSTR     x10, #10
        INSTR     x11, #11
        INSTR     x12, #12
        INSTR     x13, #13
        INSTR     x14, #14
        INSTR     x15, #15
        INSTR     x16, #16
        INSTR     x17, #17
        INSTR     x18, #18
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
