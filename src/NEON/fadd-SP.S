#define INSTR fadd
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

        fmov    v0.4s, #1.00000000
        fmov    v1.4s, #1.00000000
        fmov    v2.4s, #1.00000000
        fmov    v3.4s, #1.00000000
        fmov    v4.4s, #1.00000000
        fmov    v5.4s, #1.00000000
        fmov    v6.4s, #1.00000000
        fmov    v7.4s, #1.00000000

loop:
        subs      x4, x4, #1
        INSTR     v0.4s, v6.4s, v7.4s
        INSTR     v1.4s, v6.4s, v7.4s
        INSTR     v2.4s, v6.4s, v7.4s
        INSTR     v3.4s, v6.4s, v7.4s
        INSTR     v4.4s, v6.4s, v7.4s
        INSTR     v5.4s, v6.4s, v7.4s
        bne       loop
done:
        ret

.size latency, .-latency
