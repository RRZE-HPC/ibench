#define INSTR incd
#define NINST 16
#define N x0

.arch armv8.2-a+sve
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

        ptrue   p0.d
        fcpy    z0.d, p0/m, #1.00000000

        mov     x1, #1
        mov     x2, #2
        mov     x3, #3
        mov     x5, #4
loop:
        subs      x4, x4, #1
        INSTR    x6
        INSTR    x7
        INSTR    x8
        INSTR    x9
        INSTR    x10
        INSTR    x11
        INSTR    x12
        INSTR    x13
        INSTR    x6
        INSTR    x7
        INSTR    x8
        INSTR    x9
        INSTR    x10
        INSTR    x11
        INSTR    x12
        INSTR    x13
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
