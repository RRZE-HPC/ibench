#define INSTR add
#define NINST 6
#define N a0

.globl ninst
.data
ninst:
.long NINST
.text
.globl latency
.type latency, @function
.align 4
latency:
    # Save callee-save registers (RISC-V ABI)
    addi    sp, sp, -12*8         # Make space for 12 registers (8 bytes each)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)
    sd      s8, 64(sp)
    sd      s9, 72(sp)
    sd      s10, 80(sp)
    sd      s11, 88(sp)

    # Initialize loop counter
    li      t0, 0
    blez    N, done

loop:
    addi    t0, t0, 1           # i++

    INSTR   a3, a1, a2
    INSTR   a4, a1, a2
    INSTR   a5, a1, a2
    INSTR   a6, a1, a2
    INSTR   a7, a1, a2
    INSTR   t2, a1, a2

    blt     t0, N, loop

done:
    # Restore callee-save registers from stack
    ld      s0,  0(sp)
    ld      s1,  8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    ld      s6, 48(sp)
    ld      s7, 56(sp)
    ld      s8, 64(sp)
    ld      s9, 72(sp)
    ld      s10, 80(sp)
    ld      s11, 88(sp)
    addi    sp, sp, 12*8

    ret
.size latency, .-latency

