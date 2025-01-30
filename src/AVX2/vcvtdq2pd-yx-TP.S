#define INSTR vcvtdq2pd
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
		# expand from SSE to AVX
		vinsertf128 ymm0, ymm0, xmm0, 0x1
		# copy DP 1.0
		vmovaps xmm0, xmm0
		vmovaps xmm1, xmm0
		# Create DP 2.0
		vaddpd xmm1, xmm1, xmm1
		# Create DP 0.5
		vdivpd xmm2, xmm0, xmm1
loop:
		inc      i
		INSTR    ymm3, xmm0
		INSTR    ymm4, xmm1
		INSTR    ymm5, xmm2
		INSTR    ymm6, xmm0
		INSTR    ymm7, xmm1
		INSTR    ymm8, xmm2
		INSTR    ymm9, xmm0
		INSTR    ymm10, xmm1
		INSTR    ymm11, xmm2
		INSTR    ymm12, xmm0
		INSTR    ymm13, xmm1
		INSTR    ymm14, xmm2
		INSTR    ymm15, xmm0
		INSTR    ymm3, xmm1
		INSTR    ymm4, xmm2
		INSTR    ymm5, xmm0
		INSTR    ymm6, xmm1
		INSTR    ymm7, xmm2
		INSTR    ymm8, xmm0
		INSTR    ymm9, xmm1
		INSTR    ymm10, xmm2
		INSTR    ymm11, xmm0
		INSTR    ymm12, xmm1
		INSTR    ymm13, xmm2
		INSTR    ymm14, xmm0
		INSTR    ymm15, xmm1
		INSTR    ymm3, xmm2
		INSTR    ymm4, xmm0
		INSTR    ymm5, xmm1
		INSTR    ymm6, xmm2
		INSTR    ymm7, xmm0
		INSTR    ymm8, xmm1
		cmp      i, N
		jl       loop
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency