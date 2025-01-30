#define INSTR vgatherdpd 
#define NINST 6
#define N edi
#define i r8d


.intel_syntax noprefix
.globl ninst
.data
ninst:
.long NINST
.align 64
PI:
.long  0xffffffff, 0xffffffff, 0xffffffff,0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff  
.text
.globl latency
.type latency, @function
.align 64
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
        vpcmpeqq ymm2, ymm2, ymm2
		vpxorq xmm1, xmm1, xmm1
        kxnorb k1, k1, k1
loop:
		inc      i
		INSTR    xmm0{k1}, [rsp + xmm2*8]
		INSTR    xmm0{k1}, [rsp + xmm2*8]
		INSTR    xmm0{k1}, [rsp + xmm2*8]
		INSTR    xmm0{k1}, [rsp + xmm2*8]
		INSTR    xmm0{k1}, [rsp + xmm2*8]
		INSTR    xmm0{k1}, [rsp + xmm2*8]
		cmp      i, N
		jl       loop
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
