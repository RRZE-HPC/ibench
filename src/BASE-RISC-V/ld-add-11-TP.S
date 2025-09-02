#define INSTR ld
#define NINST 44
#define N a0

.globl ninst
.data
ninst:
.long NINST
.align 4
PI:
.long 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, \
0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, \
0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, \
0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
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

    la      t1, PI

    # Initialize loop counter
    li      t0, 0
    blez    N, done

loop:
    addi    t0, t0, 1           # i++

    INSTR   a3, 0(t1)
    add     a4, a1, a2
    INSTR   a5, 16(t1)
    add     a6, a1, a2
    INSTR   a7, 32(t1)
    add     a7, a1, a2
    INSTR   t3, 48(t1)
    add     t4, a1, a2
    INSTR   t5, 64(t1)
    add     t6, a1, a2
    INSTR   s0, 80(t1)
    add     s1, a1, a2
    INSTR   s2, 96(t1)
    add     s3, a1, a2
    INSTR   s4, 112(t1)
    add     s5, a1, a2
    INSTR   s6, 128(t1)
    add     s7, a1, a2
    INSTR   s8, 144(t1)
    add     s9, a1, a2
    INSTR   s10, 160(t1)
    add     s11, a1, a2

    INSTR   a3, 0(t1)
    add     a4, a1, a2
    INSTR   a5, 16(t1)
    add     a6, a1, a2
    INSTR   a7, 32(t1)
    add     a7, a1, a2
    INSTR   t3, 48(t1)
    add     t4, a1, a2
    INSTR   t5, 64(t1)
    add     t6, a1, a2
    INSTR   s0, 80(t1)
    add     s1, a1, a2
    INSTR   s2, 96(t1)
    add     s3, a1, a2
    INSTR   s4, 112(t1)
    add     s5, a1, a2
    INSTR   s6, 128(t1)
    add     s7, a1, a2
    INSTR   s8, 144(t1)
    add     s9, a1, a2
    INSTR   s10, 160(t1)
    add     s11, a1, a2

    INSTR   a3, 0(t1)
    add     a4, a1, a2
    INSTR   a5, 16(t1)
    add     a6, a1, a2
    INSTR   a7, 32(t1)
    add     a7, a1, a2
    INSTR   t3, 48(t1)
    add     t4, a1, a2
    INSTR   t5, 64(t1)
    add     t6, a1, a2
    INSTR   s0, 80(t1)
    add     s1, a1, a2
    INSTR   s2, 96(t1)
    add     s3, a1, a2
    INSTR   s4, 112(t1)
    add     s5, a1, a2
    INSTR   s6, 128(t1)
    add     s7, a1, a2
    INSTR   s8, 144(t1)
    add     s9, a1, a2
    INSTR   s10, 160(t1)
    add     s11, a1, a2

    INSTR   a3, 0(t1)
    add     a4, a1, a2
    INSTR   a5, 16(t1)
    add     a6, a1, a2
    INSTR   a7, 32(t1)
    add     a7, a1, a2
    INSTR   t3, 48(t1)
    add     t4, a1, a2
    INSTR   t5, 64(t1)
    add     t6, a1, a2
    INSTR   s0, 80(t1)
    add     s1, a1, a2
    INSTR   s2, 96(t1)
    add     s3, a1, a2
    INSTR   s4, 112(t1)
    add     s5, a1, a2
    INSTR   s6, 128(t1)
    add     s7, a1, a2
    INSTR   s8, 144(t1)
    add     s9, a1, a2
    INSTR   s10, 160(t1)
    add     s11, a1, a2
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

