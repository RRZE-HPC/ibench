#define INSTR sub
#define NINST 15
#define N x0

.globl ninst
.data
ninst:
.long NINST
.text
.globl latency
.type latency, @function
.align 2
latency:
    # push callee-save register onto stack

    sub     sp, sp, #64
    st1     {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
    sub     sp, sp, #64
    st1     {v12.4s, v13.4s, v14.4s, v15.4s}, [sp]
    sub     sp, sp, #64
    st1     {v16.2d, v17.2d, v18.2d, v19.2d}, [sp]
    sub     sp, sp, #64
    st1     {v20.2d, v21.2d, v22.2d, v23.2d}, [sp]
    sub     sp, sp, #64
    st1     {v24.2d, v25.2d, v26.2d, v27.2d}, [sp]
    sub     sp, sp, #64
    st1     {v28.2d, v29.2d, v30.2d, v31.2d}, [sp]
    fmov    v0.2d, #1.0
    fmov    v1.2d, #2.0
    fmov    v2.2d, #3.0
    fmov    v3.2d, #4.0
    fmov    v4.2d, #5.0
    fmov    v5.2d, #6.0
    fmov    v6.2d, #7.0
    fmov    v7.2d, #8.0
    fmov    v8.2d, #9.0
    fmov    v9.2d, #10.0
    fmov    v10.2d, #11.0
    fmov    v11.2d, #12.0
    fmov    v12.2d, #13.0
    fmov    v13.2d, #14.0
    fmov    v14.2d, #15.0
    fmov    v15.2d, #16.0

    stp     x19, x20, [sp, -96]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     x27, x28, [sp, 64]
    stp     x29, x30, [sp, 80]
    
    ptrue   p0.d
    fcpy    z0.d, p0/m, #1.00000000
    fcpy    z1.d, p0/m, #2.00000000
    fcpy    z2.d, p0/m, #3.00000000
    fcpy    z3.d, p0/m, #1.00000000
    fcpy    z4.d, p0/m, #2.00000000
    fcpy    z5.d, p0/m, #3.00000000
    fcpy    z6.d, p0/m, #1.00000000
    fcpy    z7.d, p0/m, #2.00000000

    mov     x4, N

loop:
	INSTR    z5.d, z5.d, z7.d
	INSTR    z8.d, z8.d, z10.d
	INSTR    z11.d, z11.d, z13.d
	INSTR    z14.d, z14.d, z19.d
	INSTR    z20.d, z20.d, z22.d
	INSTR    z23.d, z23.d, z25.d
	INSTR    z26.d, z26.d, z28.d
	INSTR    z0.d, z0.d, z2.d
    INSTR    z1.d, z1.d, z17.d
	INSTR    z3.d, z3.d, z18.d
	INSTR    z4.d, z4.d, z21.d
	INSTR    z6.d, z6.d, z24.d
	INSTR    z9.d, z9.d, z27.d
	INSTR    z12.d, z12.d, z29.d
	INSTR    z15.d, z15.d, z30.d

    subs    x4, x4, #1
    bne     loop
done:

    # pop callee-save registers from stack
    ldp     x19, x20, [sp]
    ldp     x21, x22, [sp, 16]
    ldp     x23, x24, [sp, 32]
    ldp     x25, x26, [sp, 48]
    ldp     x27, x28, [sp, 64]
    ldp     x29, x30, [sp, 80]
    add     sp, sp, #96

	ld1     {v28.2d, v29.2d, v30.2d, v31.2d}, [sp], #64
	ld1     {v24.2d, v25.2d, v26.2d, v27.2d}, [sp], #64
    ld1     {v20.2d, v21.2d, v22.2d, v23.2d}, [sp], #64
    ld1     {v16.2d, v17.2d, v18.2d, v19.2d}, [sp], #64
    ld1     {v12.2d, v13.2d, v14.2d, v15.2d}, [sp], #64
    ld1     {v8.2d, v9.2d, v10.2d, v11.2d}, [sp], #64


    ret
.size latency, .-latency
