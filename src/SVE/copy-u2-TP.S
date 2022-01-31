#define INSTR copy 
#define NINST 4
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
        stp x29, x30, [sp, -96]!
        stp x19, x20, [sp, 16]
        stp x21, x22, [sp, 32]
        stp x24, x25, [sp, 48]
        stp x26, x27, [sp, 64]
        str x28, [sp, 80]

        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000

        ptrue   p0.d
        ptrue   p1.d
        ptrue   p2.d
        fcpy    z1.d, p0/m, #1.00000000
        fcpy    z3.d, p0/m, #1.00000000

        mov     x20, sp
        sub     x19, sp, #64
        sub     x18, sp, #128

        sub     x17, sp, #192
        sub     x16, sp, #256
        sub     x15, sp, #320
        sub     x14, sp, #384
        sub     x13, sp, #448
        sub     x12, sp, #512
        sub     x11, sp, #578
        sub     x10, sp, #644

        mov     x1, #0
        mov     x3, #0
        mov     x7, #0
        mov     w5, #512
        mov     w6, #0

loop:
        ld1d    z0.d, p0/z, [x20, x1, lsl 3]
        st1d    z1.d, p0, [x19, x1, lsl 3]
        ld1d    z2.d, p1/z, [x20, x1, lsl 3]
        st1d    z3.d, p1, [x18, x1, lsl 3]
        add x3, x7, 8
        whilelo p2.d, w6, w5
        add x7, x3, 8
        whilelo p2.d, w6, w5

        ld1d    z4.d, p0/z, [x20, x1, lsl 3]
        st1d    z1.d, p0, [x17, x1, lsl 3]
        ld1d    z5.d, p1/z, [x20, x1, lsl 3]
        st1d    z3.d, p1, [x16, x1, lsl 3]
        add x3, x7, 8
        whilelo p2.d, w6, w5
        add x7, x3, 8
        whilelo p2.d, w6, w5

        ld1d    z6.d, p0/z, [x20, x1, lsl 3]
        st1d    z1.d, p0, [x15, x1, lsl 3]
        ld1d    z7.d, p1/z, [x20, x1, lsl 3]
        st1d    z3.d, p1, [x14, x1, lsl 3]
        add x3, x7, 8
        whilelo p2.d, w6, w5
        add x7, x3, 8
        whilelo p2.d, w6, w5

        ld1d    z8.d, p0/z, [x20, x1, lsl 3]
        st1d    z1.d, p0, [x13, x1, lsl 3]
        ld1d    z9.d, p1/z, [x20, x1, lsl 3]
        st1d    z3.d, p1, [x12, x1, lsl 3]
        add x3, x7, 8
        whilelo p2.d, w6, w5
        add x7, x3, 8
        whilelo p2.d, w6, w5

        subs      x4, x4, #1
        bne       loop
done:
        # pop callee-save registers from stack
        ldp x19, x20, [sp, 16]
        ldp x21, x22, [sp, 32]
        ldp x24, x25, [sp, 48]
        ldp x26, x27, [sp, 64]
        ldr x28, [sp, 80]
        ldp x29, x30, [sp], 96
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
