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

        mov     x1, #-64
        mov     x2, #-128 
        mov     x3, #-192
        mov     x5, #-256
        mov     x6, #-320
        mov     x7, #-384
loop:
        subs      x4, x4, #1
        INSTR    {z0.d}, p0, [sp, x1, lsl 3]
        INSTR    {z1.d}, p0, [sp, x2, lsl 3]
        INSTR    {z2.d}, p0, [sp, x3, lsl 3]
        INSTR    {z3.d}, p0, [sp, x5, lsl 3]
        INSTR    {z4.d}, p0, [sp, x6, lsl 3]
        INSTR    {z5.d}, p0, [sp, x7, lsl 3] 
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
