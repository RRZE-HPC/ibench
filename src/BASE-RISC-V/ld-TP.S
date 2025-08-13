#define INSTR ld
#define NINST 22
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
    INSTR   a4, 8(t1)
    INSTR   a5, 16(t1)
    INSTR   a6, 24(t1)
    INSTR   a7, 32(t1)
    INSTR   t2, 40(t1)
    INSTR   t3, 48(t1)
    INSTR   t4, 56(t1)
    INSTR   t5, 64(t1)
    INSTR   t6, 72(t1)
    INSTR   s0, 80(t1)
    INSTR   s1, 88(t1)
    INSTR   s2, 96(t1)
    INSTR   s3, 104(t1)
    INSTR   s4, 112(t1)
    INSTR   s5, 120(t1)
    INSTR   s6, 128(t1)
    INSTR   s7, 136(t1)
    INSTR   s8, 142(t1)
    INSTR   s9, 150(t1)
    INSTR   s10, 158(t1)
    INSTR   s11, 164(t1)

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

