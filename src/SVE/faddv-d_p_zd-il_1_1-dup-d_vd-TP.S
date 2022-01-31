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

loop:
        subs      x4, x4, #1
        INSTR     d2, p0, z0.d 
        dup       d3, v1.d[0]
        INSTR     d4, p0, z0.d 
        dup       d5, v1.d[0]
        INSTR     d6, p0, z1.d 
        dup       d7, v1.d[0]
        INSTR     d8, p0, z0.d 
        dup       d9, v1.d[0]
        INSTR     d10, p0, z1.d
        dup       d11, v1.d[0]
        INSTR     d12, p0, z0.d 
        dup       d13, v1.d[0]
        INSTR     d14, p0, z1.d 
        dup       d15, v1.d[0]
        INSTR     d16, p0, z0.d 
        dup       d17, v1.d[0]
        INSTR     d18, p0, z1.d 
        dup       d19, v1.d[0]
        INSTR     d20, p0, z0.d 
        dup       d21, v1.d[0]
        INSTR     d22, p0, z1.d
        dup       d23, v1.d[0]
        INSTR     d24, p0, z0.d 
        dup       d25, v1.d[0]
        INSTR     d26, p0, z1.d 
        dup       d27, v1.d[0]
        INSTR     d28, p0, z0.d 
        dup       d29, v1.d[0]
        INSTR     d30, p0, z1.d 
        dup       d2, v1.d[0]
        INSTR     d3, p0, z0.d 
        dup       d4, v1.d[0]
        INSTR     d5, p0, z1.d
        dup       d6, v1.d[0]
        INSTR     d7, p0, z0.d 
        dup       d8, v1.d[0]
        INSTR     d9, p0, z1.d 
        dup       d10, v1.d[0]
        INSTR     d11, p0, z0.d 
        dup       d12, v1.d[0]
        INSTR     d13, p0, z1.d 
        dup       d14, v1.d[0]
        INSTR     d15, p0, z0.d 
        dup       d16, v1.d[0]
        INSTR     d17, p0, z1.d
        dup       d18, v1.d[0]
        INSTR     d19, p0, z0.d 
        dup       d20, v1.d[0]
        INSTR     d21, p0, z1.d 
        dup       d22, v1.d[0]
        INSTR     d23, p0, z0.d 
        dup       d24, v1.d[0]
        INSTR     d25, p0, z1.d 
        dup       d26, v1.d[0]
        INSTR     d27, p0, z0.d 
        dup       d28, v1.d[0]
        INSTR     d29, p0, z1.d 
        dup       d30, v1.d[0]
        INSTR     d2, p0, z0.d 
        dup       d3, v1.d[0]
        INSTR     d4, p0, z1.d 
        dup       d5, v1.d[0]
        INSTR     d6, p0, z0.d 
        dup       d7, v1.d[0]
        INSTR     d8, p0, z1.d
        dup       d9, v1.d[0]
        INSTR     d10, p0, z0.d 
        dup       d11, v1.d[0]
        INSTR     d12, p0, z1.d 
        dup       d13, v1.d[0]
        INSTR     d14, p0, z0.d 
        dup       d15, v1.d[0]
        INSTR     d16, p0, z1.d 
        dup       d17, v1.d[0]
        INSTR     d18, p0, z0.d 
        dup       d19, v1.d[0]
        INSTR     d20, p0, z1.d
        dup       d21, v1.d[0]
        INSTR     d22, p0, z0.d 
        dup       d23, v1.d[0]
        INSTR     d24, p0, z1.d 
        dup       d25, v1.d[0]
        INSTR     d26, p0, z0.d 
        dup       d27, v1.d[0]
        INSTR     d28, p0, z1.d 
        dup       d29, v1.d[0]
        INSTR     d30, p0, z0.d 
        dup       d2, v1.d[0]
        INSTR     d3, p0, z1.d
        dup       d4, v1.d[0]
        INSTR     d5, p0, z0.d 
        dup       d6, v1.d[0]
        INSTR     d7, p0, z1.d 
        dup       d8, v1.d[0]
        INSTR     d9, p0, z0.d 
        dup       d10, v1.d[0]
       
        bne       loop
done:
        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
