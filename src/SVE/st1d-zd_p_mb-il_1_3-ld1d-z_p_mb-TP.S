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
        ptrue   p1.d
        fcpy    z0.d, p0/m, #1.00000000
        fcpy    z1.d, p0/m, #1.00000000
        fcpy    z2.d, p0/m, #1.00000000

        mov     x1, sp
        mov     x2, sp
        mov     x3, sp
        mov     x5, sp
        mov     x6, sp
        mov     x7, sp
        sub     x1, x1, #64
        sub     x2, x2, #128
        sub     x3, x3, #192
        sub     x5, x5, #256
        sub     x6, x6, #320
        sub     x7, x7, #384
loop:
        subs      x4, x4, #1
        INSTR    {z0.d}, p0, [sp]
        ld1d     {z1.d}, p1/z, [x7]
        ld1d     {z2.d}, p1/z, [x7]
        ld1d     {z3.d}, p1/z, [x7]
        INSTR    {z4.d}, p0, [x1]
        ld1d     {z5.d}, p1/z, [x7]
        ld1d     {z6.d}, p1/z, [x7]
        ld1d     {z7.d}, p1/z, [x7]
        INSTR    {z8.d}, p0, [x2]
        ld1d     {z9.d}, p1/z, [x7]
        ld1d     {z10.d}, p1/z, [x7]
        ld1d     {z11.d}, p1/z, [x7]
        INSTR    {z12.d}, p0, [x3]
        ld1d     {z13.d}, p1/z, [x7]
        ld1d     {z14.d}, p1/z, [x7]
        ld1d     {z15.d}, p1/z, [x7]
        INSTR    {z16.d}, p0, [x5]
        ld1d     {z17.d}, p1/z, [x7]
        ld1d     {z18.d}, p1/z, [x7]
        ld1d     {z19.d}, p1/z, [x7]
        INSTR    {z20.d}, p0, [x6] 
        ld1d     {z21.d}, p1/z, [x7]
        ld1d     {z22.d}, p1/z, [x7]
        ld1d     {z23.d}, p1/z, [x7]
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
