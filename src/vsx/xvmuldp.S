#define INSTR xvmuldp
#define NINST 6
#define N 3

.globl ninst
.data
ninst:
.long NINST
.align 16
zero:
.double 0.0, 0.0
one:
.double 1.0, 1.0
.text
.abiversion 2
.section    ".toc","aw"
.section    ".text"
.align 2
.globl latency
.type latency, @function
latency :
0:      addis 2,12,.TOC.-0b@ha
        addi 2,2,.TOC.-0b@l
.localentry latency, .-latency

        mtctr N           # move to count register
        li 10, 0                # offset zero
        addis 9,2,one@toc@ha   # upper 32 bit of address
        addi 9,9,one@toc@l     # lower 32 bit of address
        lxvd2x 0, 0, 9
        addis 9,2,one@toc@ha    # upper 32 bit of address
        addi 9,9,one@toc@l      # lower 32 bit of address
        lxvd2x 1, 0, 9
loop:
        INSTR     0, 0, 1
        INSTR     0, 0, 1
        INSTR     0, 0, 1
        INSTR     0, 0, 1
        INSTR     0, 0, 1
        INSTR     0, 0, 1
        bdnz loop
        xvmovdp 1, 0
        blr 
.size latency, .-latency
