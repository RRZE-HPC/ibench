#define INSTR fadda
#define NINST 30
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
        INSTR     s1, p0, s1, z0.s 
        INSTR     s2, p0, s2, z0.s 
        INSTR     s3, p0, s3, z0.s 
        INSTR     s4, p0, s4, z0.s 
        INSTR     s5, p0, s5, z0.s 
        INSTR     s6, p0, s6, z0.s 
        INSTR     s7, p0, s7, z0.s
        INSTR     s8, p0, s8, z0.s 
        INSTR     s9, p0, s9, z0.s 
        INSTR     s10, p0, s10, z0.s 
        INSTR     s11, p0, s11, z0.s 
        INSTR     s12, p0, s12, z0.s 
        INSTR     s13, p0, s13, z0.s
        INSTR     s14, p0, s14, z0.s 
        INSTR     s15, p0, s15, z0.s 
        INSTR     s16, p0, s16, z0.s 
        INSTR     s17, p0, s17, z0.s 
        INSTR     s18, p0, s18, z0.s 
        INSTR     s19, p0, s19, z0.s
        INSTR     s21, p0, s21, z0.s 
        INSTR     s22, p0, s22, z0.s 
        INSTR     s23, p0, s23, z0.s 
        INSTR     s24, p0, s24, z0.s 
        INSTR     s25, p0, s25, z0.s 
        INSTR     s26, p0, s26, z0.s
        INSTR     s27, p0, s27, z0.s 
        INSTR     s28, p0, s28, z0.s 
        INSTR     s29, p0, s29, z0.s 
        INSTR     s30, p0, s30, z0.s 
        bne       loop
sone:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
