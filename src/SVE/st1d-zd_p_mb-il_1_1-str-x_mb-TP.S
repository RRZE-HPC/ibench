#define INSTR st1d
#define NINST 6
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

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000

        ptrue   p0.d
        fcpy    z0.d, p0/m, #1.00000000
        fcpy    z1.d, p0/m, #1.00000000
        fcpy    z2.d, p0/m, #1.00000000

        mov     x1, sp
        mov     x2, sp
        mov     x3, sp
        mov     x5, sp
        mov     x6, sp
        sub     x1, x1, #64
        sub     x2, x2, #128
        sub     x3, x3, #192
        sub     x5, x5, #256
        sub     x6, x6, #320
loop:
        subs      x4, x4, #1
        INSTR    {z0.d}, p0, [sp]
        str      x7, [sp]
        INSTR    {z1.d}, p0, [x1]
        str      x8, [sp]
        INSTR    {z2.d}, p0, [x2]
        str      x9, [sp]
        INSTR    {z3.d}, p0, [x3]
        str      x10, [sp]
        INSTR    {z4.d}, p0, [x5]
        str      x11, [sp]
        INSTR    {z5.d}, p0, [x6] 
        str      x12, [sp]
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
