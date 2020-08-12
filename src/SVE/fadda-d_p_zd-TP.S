#define INSTR fadda
#define NINST 180
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
        fcpy    z1.d, p0/m, #2.00000000
        fcpy    z2.d, p0/m, #3.00000000
        fcpy    z3.d, p0/m, #1.00000000
        fcpy    z4.d, p0/m, #2.00000000
        fcpy    z5.d, p0/m, #3.00000000
        fcpy    z6.d, p0/m, #1.00000000
        fcpy    z7.d, p0/m, #2.00000000

loop:
        subs      x4, x4, #1
        INSTR     d1, p0, d1, z0.d 
        INSTR     d2, p0, d2, z0.d 
        INSTR     d3, p0, d3, z0.d 
        INSTR     d4, p0, d4, z0.d 
        INSTR     d5, p0, d5, z0.d 
        INSTR     d6, p0, d6, z0.d 
        INSTR     d7, p0, d7, z0.d
        INSTR     d8, p0, d8, z0.d 
        INSTR     d9, p0, d9, z0.d 
        INSTR     d10, p0, d10, z0.d 
        INSTR     d11, p0, d11, z0.d 
        INSTR     d12, p0, d12, z0.d 
        INSTR     d13, p0, d13, z0.d
        INSTR     d14, p0, d14, z0.d 
        INSTR     d15, p0, d15, z0.d 
        INSTR     d16, p0, d16, z0.d 
        INSTR     d17, p0, d17, z0.d 
        INSTR     d18, p0, d18, z0.d 
        INSTR     d19, p0, d19, z0.d
        INSTR     d21, p0, d21, z0.d 
        INSTR     d22, p0, d22, z0.d 
        INSTR     d23, p0, d23, z0.d 
        INSTR     d24, p0, d24, z0.d 
        INSTR     d25, p0, d25, z0.d 
        INSTR     d26, p0, d26, z0.d
        INSTR     d27, p0, d27, z0.d 
        INSTR     d28, p0, d28, z0.d 
        INSTR     d29, p0, d29, z0.d 
        INSTR     d30, p0, d30, z0.d 
        INSTR     d1, p0, d1, z0.d 
        INSTR     d2, p0, d2, z0.d 
        INSTR     d3, p0, d3, z0.d 
        INSTR     d4, p0, d4, z0.d 
        INSTR     d5, p0, d5, z0.d 
        INSTR     d6, p0, d6, z0.d 
        INSTR     d7, p0, d7, z0.d
        INSTR     d8, p0, d8, z0.d 
        INSTR     d9, p0, d9, z0.d 
        INSTR     d10, p0, d10, z0.d 
        INSTR     d11, p0, d11, z0.d 
        INSTR     d12, p0, d12, z0.d 
        INSTR     d13, p0, d13, z0.d
        INSTR     d14, p0, d14, z0.d 
        INSTR     d15, p0, d15, z0.d 
        INSTR     d16, p0, d16, z0.d 
        INSTR     d17, p0, d17, z0.d 
        INSTR     d18, p0, d18, z0.d 
        INSTR     d19, p0, d19, z0.d
        INSTR     d21, p0, d21, z0.d 
        INSTR     d22, p0, d22, z0.d 
        INSTR     d23, p0, d23, z0.d 
        INSTR     d24, p0, d24, z0.d 
        INSTR     d25, p0, d25, z0.d 
        INSTR     d26, p0, d26, z0.d
        INSTR     d27, p0, d27, z0.d 
        INSTR     d28, p0, d28, z0.d 
        INSTR     d29, p0, d29, z0.d 
        INSTR     d30, p0, d30, z0.d
        
        INSTR     d1, p0, d1, z0.d 
        INSTR     d2, p0, d2, z0.d 
        INSTR     d3, p0, d3, z0.d 
        INSTR     d4, p0, d4, z0.d 
        INSTR     d5, p0, d5, z0.d 
        INSTR     d6, p0, d6, z0.d 
        INSTR     d7, p0, d7, z0.d
        INSTR     d8, p0, d8, z0.d 
        INSTR     d9, p0, d9, z0.d 
        INSTR     d10, p0, d10, z0.d 
        INSTR     d11, p0, d11, z0.d 
        INSTR     d12, p0, d12, z0.d 
        INSTR     d13, p0, d13, z0.d
        INSTR     d14, p0, d14, z0.d 
        INSTR     d15, p0, d15, z0.d 
        INSTR     d16, p0, d16, z0.d 
        INSTR     d17, p0, d17, z0.d 
        INSTR     d18, p0, d18, z0.d 
        INSTR     d19, p0, d19, z0.d
        INSTR     d21, p0, d21, z0.d 
        INSTR     d22, p0, d22, z0.d 
        INSTR     d23, p0, d23, z0.d 
        INSTR     d24, p0, d24, z0.d 
        INSTR     d25, p0, d25, z0.d 
        INSTR     d26, p0, d26, z0.d
        INSTR     d27, p0, d27, z0.d 
        INSTR     d28, p0, d28, z0.d 
        INSTR     d29, p0, d29, z0.d 
        INSTR     d30, p0, d30, z0.d 
        INSTR     d1, p0, d1, z0.d 
        INSTR     d2, p0, d2, z0.d 
        INSTR     d3, p0, d3, z0.d 
        INSTR     d4, p0, d4, z0.d 
        INSTR     d5, p0, d5, z0.d 
        INSTR     d6, p0, d6, z0.d 
        INSTR     d7, p0, d7, z0.d
        INSTR     d8, p0, d8, z0.d 
        INSTR     d9, p0, d9, z0.d 
        INSTR     d10, p0, d10, z0.d 
        INSTR     d11, p0, d11, z0.d 
        INSTR     d12, p0, d12, z0.d 
        INSTR     d13, p0, d13, z0.d
        INSTR     d14, p0, d14, z0.d 
        INSTR     d15, p0, d15, z0.d 
        INSTR     d16, p0, d16, z0.d 
        INSTR     d17, p0, d17, z0.d 
        INSTR     d18, p0, d18, z0.d 
        INSTR     d19, p0, d19, z0.d
        INSTR     d21, p0, d21, z0.d 
        INSTR     d22, p0, d22, z0.d 
        INSTR     d23, p0, d23, z0.d 
        INSTR     d24, p0, d24, z0.d 
        INSTR     d25, p0, d25, z0.d 
        INSTR     d26, p0, d26, z0.d
        INSTR     d27, p0, d27, z0.d 
        INSTR     d28, p0, d28, z0.d 
        INSTR     d29, p0, d29, z0.d 
        INSTR     d30, p0, d30, z0.d
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
