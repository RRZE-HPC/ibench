#define INSTR vpsrlw
#define ALTI vpsrlw
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
loop:
		inc      i
		INSTR    xmm2, xmm1, xmm0  
        ALTI     xmm3, xmm0, 3 
		INSTR    xmm4, xmm1, xmm0 
        ALTI     xmm5, xmm0, 3
        INSTR    xmm6, xmm1, xmm0 
        ALTI     xmm7, xmm0, 3
		INSTR    xmm8, xmm1, xmm0 
        ALTI     xmm9, xmm0, 3
		INSTR    xmm10, xmm1, xmm0 
        ALTI     xmm11, xmm0, 3
		INSTR    xmm12, xmm1, xmm0 
		ALTI     xmm13, xmm0, 3 
		#INSTR    xmm14, xmm1
		#ALTI     xmm15, xmm0, xmm0 
		cmp      i, N
		jl       loop
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
