#define INSTR ld1d
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
        add     x1, x1, #512
        add     x2, x1, #512
        sub     x3, x2, #512
        sub     x5, x3, #512
        sub     x6, x5, #512
loop:
        subs      x4, x4, #1
        INSTR    {z0.d}, p0/z, [sp]
        INSTR    {z1.d}, p0/z, [x1]
        INSTR    {z2.d}, p0/z, [x2]
        INSTR    {z3.d}, p0/z, [x3]
        INSTR    {z4.d}, p0/z, [x5]
        INSTR    {z5.d}, p0/z, [x6] 
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
