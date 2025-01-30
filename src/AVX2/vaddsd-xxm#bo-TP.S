#define INSTR vaddsd
#define NINST 32
#define N edi
#define i r8d


.intel_syntax noprefix
.globl ninst
.data
ninst:
.long NINST
.align 32
PI:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
.text
.globl latency
.type latency, @function
.align 32
latency:
		push      rbp
		mov       rbp, rsp
		xor       i, i
		test      N, N
		jle       done
		# create DP 1.0
		vpcmpeqw xmm0, xmm0, xmm0		# all ones
		vpsllq xmm0, xmm0, 54			# logical left shift: 11111110..0 (54=64-(10-1))
		vpsrlq xmm0, xmm0, 2			# logical right shift: 1 bit for sign; leading mantissa bit is zero
loop:
		inc      i
		INSTR    xmm3, xmm0, [8+rip]
		INSTR    xmm4, xmm1, [8+rip]
		INSTR    xmm5, xmm2, [8+rip]
		INSTR    xmm6, xmm0, [8+rip]
		INSTR    xmm7, xmm1, [8+rip]
		INSTR    xmm8, xmm2, [8+rip]
		INSTR    xmm9, xmm0, [8+rip]
		INSTR    xmm10, xmm1, [8+rip]
		INSTR    xmm11, xmm2, [8+rip]
		INSTR    xmm12, xmm0, [8+rip]
		INSTR    xmm13, xmm1, [8+rip]
		INSTR    xmm14, xmm2, [8+rip]
		INSTR    xmm15, xmm0, [8+rip]
		INSTR    xmm3, xmm1, [8+rip]
		INSTR    xmm4, xmm2, [8+rip]
		INSTR    xmm5, xmm0, [8+rip]
		INSTR    xmm6, xmm1, [8+rip]
		INSTR    xmm7, xmm2, [8+rip]
		INSTR    xmm8, xmm0, [8+rip]
		INSTR    xmm9, xmm1, [8+rip]
		INSTR    xmm10, xmm2, [8+rip]
		INSTR    xmm11, xmm0, [8+rip]
		INSTR    xmm12, xmm1, [8+rip]
		INSTR    xmm13, xmm2, [8+rip]
		INSTR    xmm14, xmm0, [8+rip]
		INSTR    xmm15, xmm1, [8+rip]
		INSTR    xmm3, xmm2, [8+rip]
		INSTR    xmm4, xmm0, [8+rip]
		INSTR    xmm5, xmm1, [8+rip]
		INSTR    xmm6, xmm2, [8+rip]
		INSTR    xmm7, xmm0, [8+rip]
		INSTR    xmm8, xmm1, [8+rip]
		cmp      i, N
		jl       loop
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
