#define INSTR vadd.vv
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

    # ------------------------------------------------------
    # Configure the Vector Unit
    # li a1, 4       -> Load immediate value 4 into a1
    # vsetvli t0,a1,e32,m1,ta,ma
    #   - Sets VL=4 (4 vector elements active)
    #   - e32 = 32-bit elements
    #   - m1  = LMUL=1 (no grouping of registers)
    #   - ta,ma = tail-agnostic, mask-agnostic
    # ------------------------------------------------------
    li      a1, 4
    vsetvli t0, a1, e64, m1, ta, ma

    vmv.v.i v0, 1         # v0 = {1,1}
    vmv.v.i v1, 2         # v1 = {2,2}

    # Initialize loop counter
    li      t0, 0
    blez    N, done

loop:
    addi    t0, t0, 1           # i++

    INSTR   v1, v0, v0
    INSTR   v2, v0, v0
    INSTR   v3, v0, v0
    INSTR   v4, v0, v0
    INSTR   v5, v0, v0
    INSTR   v6, v0, v0

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

