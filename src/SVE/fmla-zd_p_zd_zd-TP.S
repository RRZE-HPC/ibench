#define INSTR fmla
#define NINST 24
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

loop:
        subs      x4, x4, #1
        INSTR     z1.d, p0/m, z0.d, z0.d 
        INSTR     z2.d, p0/m, z0.d, z0.d 
        INSTR     z3.d, p0/m, z0.d, z0.d 
        INSTR     z4.d, p0/m, z0.d, z0.d 
        INSTR     z5.d, p0/m, z0.d, z0.d 
        INSTR     z6.d, p0/m, z0.d, z0.d 
        INSTR     z7.d, p0/m, z0.d, z0.d 
        INSTR     z8.d, p0/m, z0.d, z0.d 
        INSTR     z9.d, p0/m, z0.d, z0.d 
        INSTR     z10.d, p0/m, z0.d, z0.d 
        INSTR     z11.d, p0/m, z0.d, z0.d 
        INSTR     z12.d, p0/m, z0.d, z0.d 
        INSTR     z13.d, p0/m, z0.d, z0.d
        INSTR     z14.d, p0/m, z0.d, z0.d 
        INSTR     z15.d, p0/m, z0.d, z0.d 
        INSTR     z16.d, p0/m, z0.d, z0.d 
        INSTR     z17.d, p0/m, z0.d, z0.d 
        INSTR     z18.d, p0/m, z0.d, z0.d
        INSTR     z19.d, p0/m, z0.d, z0.d
        INSTR     z20.d, p0/m, z0.d, z0.d 
        INSTR     z21.d, p0/m, z0.d, z0.d 
        INSTR     z22.d, p0/m, z0.d, z0.d 
        INSTR     z23.d, p0/m, z0.d, z0.d 
        INSTR     z24.d, p0/m, z0.d, z0.d
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
