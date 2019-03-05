#define INSTR fsub
#define NINST 6
# ABI defines r0-r3 as registers in which parameters are passed to functions
# These registers can also be used as scratch registers.
#define N r0
#define i r1

.globl ninst
.data
ninst:
.long NINST
.text
.globl latency
.type latency, @function
.align 2
latency:

        mov     x4, x0

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000
        fmov    v3.2d, #1.00000000
        fmov    v4.2d, #1.00000000
        fmov    v5.2d, #1.00000000
        fmov    v6.2d, #1.00000000
        fmov    v7.2d, #1.00000000

loop:
        subs      x4, x4, #1
        INSTR     v0.2d, v6.2d, v7.2d
        INSTR     v1.2d, v6.2d, v7.2d
        INSTR     v2.2d, v6.2d, v7.2d
        INSTR     v3.2d, v6.2d, v7.2d
        INSTR     v4.2d, v6.2d, v7.2d
        INSTR     v5.2d, v6.2d, v7.2d
        bne       loop
done:
        ret

.size latency, .-latency
