#define INSTR mla
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
        INSTR     z1.s, p0/m, z0.s, z0.s 
        INSTR     z2.s, p0/m, z0.s, z0.s 
        INSTR     z3.s, p0/m, z0.s, z0.s 
        INSTR     z4.s, p0/m, z0.s, z0.s 
        INSTR     z5.s, p0/m, z0.s, z0.s 
        INSTR     z6.s, p0/m, z0.s, z0.s 
        INSTR     z7.s, p0/m, z0.s, z0.s 
        INSTR     z8.s, p0/m, z0.s, z0.s 
        INSTR     z9.s, p0/m, z0.s, z0.s 
        INSTR     z10.s, p0/m, z0.s, z0.s 
        INSTR     z11.s, p0/m, z0.s, z0.s 
        INSTR     z12.s, p0/m, z0.s, z0.s 
        INSTR     z13.s, p0/m, z0.s, z0.s
        INSTR     z14.s, p0/m, z0.s, z0.s 
        INSTR     z15.s, p0/m, z0.s, z0.s 
        INSTR     z16.s, p0/m, z0.s, z0.s 
        INSTR     z17.s, p0/m, z0.s, z0.s 
        INSTR     z18.s, p0/m, z0.s, z0.s
        INSTR     z19.s, p0/m, z0.s, z0.s
        INSTR     z20.s, p0/m, z0.s, z0.s 
        INSTR     z21.s, p0/m, z0.s, z0.s 
        INSTR     z22.s, p0/m, z0.s, z0.s 
        INSTR     z23.s, p0/m, z0.s, z0.s 
        INSTR     z24.s, p0/m, z0.s, z0.s
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
