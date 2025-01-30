#define INSTR vmovupd
#define NINST 16
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
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1
		# copy DP 1.0
		vmovaps zmm0, zmm0
		vmovaps zmm1, zmm0
		# Create DP 2.0
		vaddpd zmm1, zmm1, zmm1
		# Create DP 0.5
		vdivpd zmm2, zmm0, zmm1
loop:
		inc      i
		INSTR    zmm0, [rip]
		INSTR    zmm1, [rip+64]
		INSTR    zmm2, [rip+128]
		INSTR    zmm3, [rip+192]
		INSTR    zmm4, [rip+256]
		INSTR    zmm5, [rip+320]
		INSTR    zmm6, [rip+384]
		INSTR    zmm7, [rip+448]
		INSTR    zmm8, [rip+512]
		INSTR    zmm9, [rip+576]
		INSTR    zmm10, [rip+640]
		INSTR    zmm11, [rip+704]
		INSTR    zmm12, [rip+768]
		INSTR    zmm13, [rip+832]
		INSTR    zmm14, [rip+896]
		INSTR    zmm15, [rip+960]
		INSTR    zmm16, [rip+1024]
		cmp      i, N
		jl       loop
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
