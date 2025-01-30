#define INSTR vcvtpd2ps
#define ALTI vmulpd 
#define NINST 6
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
		vmovaps ymm0, ymm0
		vmovaps ymm1, ymm0
		# Create DP 2.0
		vaddpd ymm1, ymm1, ymm1
		# Create DP 0.5
		vdivpd ymm2, ymm0, ymm1
        push r12
        push r13
        push r14
        push r15
loop:
		inc      i
		INSTR    xmm2, ymm1
        #ALTI     xmm3, xmm0, xmm0
        #vmovupd  [rip+PI], xmm15
		INSTR    xmm4, ymm1
        #ALTI     xmm5, xmm0, xmm0
        #vmovupd  [rip+PI], xmm15
		INSTR    xmm6, ymm1
        #ALTI     xmm7, xmm0, xmm0
        #vmovupd  [rip+PI], xmm15
		INSTR    xmm8, ymm1
        #ALTI     xmm9, xmm0, xmm0
        #vmovupd  [rip+PI], xmm15
		INSTR    xmm10, ymm1
        #ALTI     xmm11, xmm0, xmm0
        #vmovupd  [rip+PI], xmm15
		INSTR    xmm12, ymm1
		#ALTI     xmm13, xmm0, xmm0 
        #vmovupd  [rip+PI], xmm15
		#INSTR    xmm14, ymm1
		#ALTI     xmm15, xmm0, xmm0 
		cmp      i, N
		jl       loop
done:
        pop r15
        pop r14
        pop r13
        pop r12
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
