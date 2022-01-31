#define INSTR faddv
#define NINST 48
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

loop:
        subs      x4, x4, #1
        INSTR     d2, p0, z0.d 
        fadd      z3.d, z1.d, z1.d
        INSTR     d4, p0, z0.d 
        fadd      z5.d, z1.d, z1.d
        INSTR     d6, p0, z0.d 
        fadd      z7.d, z1.d, z1.d
        INSTR     d8, p0, z0.d 
        fadd      z9.d, z1.d, z1.d
        INSTR     d10, p0, z0.d
        fadd      z11.d, z1.d, z1.d
        INSTR     d12, p0, z0.d 
        fadd      z13.d, z1.d, z1.d
        INSTR     d14, p0, z0.d 
        fadd      z15.d, z1.d, z1.d
        INSTR     d16, p0, z0.d 
        fadd      z17.d, z1.d, z1.d
        INSTR     d18, p0, z0.d 
        fadd      z19.d, z1.d, z1.d
        INSTR     d20, p0, z0.d 
        fadd      z21.d, z1.d, z1.d
        INSTR     d22, p0, z0.d
        fadd      z23.d, z1.d, z1.d
        INSTR     d24, p0, z0.d 
        fadd      z25.d, z1.d, z1.d
        INSTR     d26, p0, z0.d 
        fadd      z27.d, z1.d, z1.d
        INSTR     d28, p0, z0.d 
        fadd      z29.d, z1.d, z1.d
        INSTR     d30, p0, z0.d 
        fadd      z2.d, z1.d, z1.d
        INSTR     d3, p0, z0.d 
        fadd      z4.d, z1.d, z1.d
        INSTR     d5, p0, z0.d
        fadd      z6.d, z1.d, z1.d
        INSTR     d7, p0, z0.d 
        fadd      z8.d, z1.d, z1.d
        INSTR     d9, p0, z0.d 
        fadd      z10.d, z1.d, z1.d
        INSTR     d11, p0, z0.d 
        fadd      z12.d, z1.d, z1.d
        INSTR     d13, p0, z0.d 
        fadd      z14.d, z1.d, z1.d
        INSTR     d15, p0, z0.d 
        fadd      z16.d, z1.d, z1.d
        INSTR     d17, p0, z0.d
        fadd      z18.d, z1.d, z1.d
        INSTR     d19, p0, z0.d 
        fadd      z20.d, z1.d, z1.d
        INSTR     d21, p0, z0.d 
        fadd      z22.d, z1.d, z1.d
        INSTR     d23, p0, z0.d 
        fadd      z24.d, z1.d, z1.d
        INSTR     d25, p0, z0.d 
        fadd      z26.d, z1.d, z1.d
        INSTR     d27, p0, z0.d 
        fadd      z28.d, z1.d, z1.d
        INSTR     d29, p0, z0.d 
        fadd      z30.d, z1.d, z1.d
        INSTR     d2, p0, z0.d 
        fadd      z3.d, z1.d, z1.d
        INSTR     d4, p0, z0.d 
        fadd      z5.d, z1.d, z1.d
        INSTR     d6, p0, z0.d 
        fadd      z7.d, z1.d, z1.d
        INSTR     d8, p0, z0.d
        fadd      z9.d, z1.d, z1.d
        INSTR     d10, p0, z0.d 
        fadd      z11.d, z1.d, z1.d
        INSTR     d12, p0, z0.d 
        fadd      z13.d, z1.d, z1.d
        INSTR     d14, p0, z0.d 
        fadd      z15.d, z1.d, z1.d
        INSTR     d16, p0, z0.d 
        fadd      z17.d, z1.d, z1.d
        INSTR     d18, p0, z0.d 
        fadd      z19.d, z1.d, z1.d
        INSTR     d20, p0, z0.d
        fadd      z21.d, z1.d, z1.d
        INSTR     d22, p0, z0.d 
        fadd      z23.d, z1.d, z1.d
        INSTR     d24, p0, z0.d 
        fadd      z25.d, z1.d, z1.d
        INSTR     d26, p0, z0.d 
        fadd      z27.d, z1.d, z1.d
        INSTR     d28, p0, z0.d 
        fadd      z29.d, z1.d, z1.d
        INSTR     d30, p0, z0.d 
        fadd      z2.d, z1.d, z1.d
        INSTR     d3, p0, z0.d
        fadd      z4.d, z1.d, z1.d
        INSTR     d5, p0, z0.d 
        fadd      z6.d, z1.d, z1.d
        INSTR     d7, p0, z0.d 
        fadd      z8.d, z1.d, z1.d
        INSTR     d9, p0, z0.d 
        fadd      z10.d, z1.d, z1.d
       
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
