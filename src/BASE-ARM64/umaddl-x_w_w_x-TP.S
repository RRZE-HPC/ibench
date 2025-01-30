#define INSTR umaddl
#define NINST 8
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

    stp     x19, x20, [sp, -96]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     x27, x28, [sp, 64]
    stp     x29, x30, [sp, 80]
    
    mov     x4, N

loop:
	INSTR    x5, w6, w7, x8
	INSTR    x9, w10, w11, x12
	INSTR    x13, w14, w15, x19
	INSTR    x20, w21, w22, x23
	INSTR    x24, w25, w26, x27
	INSTR    x28, w5, w6, x7
	INSTR    x8, w9, w10, x11
	INSTR    x12, w13, w14, x15

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


    ret
.size latency, .-latency