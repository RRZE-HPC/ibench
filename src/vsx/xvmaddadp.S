#define INSTR xvmaddadp
#define NINST 6
#define N 3

.globl ninst
.data
ninst:
.long NINST
.align 16
zero:
.double 0.0, 0.0
two:
.double 2.0, 2.0
three:
.double 3.0, 3.0
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
        # load DP FP zero
        li 10, 0

        addis 9,2,zero@toc@ha
        addi 9,9,zero@toc@l
        lxvd2x 0, 0, 9

        addis 9,2,two@toc@ha
        addi 9,9,two@toc@l
        lxvd2x 1, 0, 9

        addis 9,2,three@toc@ha
        addi 9,9,three@toc@l
        lxvd2x 2, 0, 9
loop:
        INSTR     0, 1, 2
        INSTR     0, 1, 2
        INSTR     0, 1, 2
        INSTR     0, 1, 2
        INSTR     0, 1, 2
        INSTR     0, 1, 2
        bdnz loop
        xvmovdp 1, 0
        blr 
.size latency, .-latency
