#define INSTR vmovupd
#define NINST 2
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
IP:
.long  0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000
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
        mov       rbp, rbp
        push r12
        push r13
        push r14
        push r15
loop:
		inc      i
        #vmovupd      ymm10, [rip+PI]
		INSTR    ymm3, [rip+PI]
		INSTR    ymm4, [rip+PI]
		#INSTR    [rip+IP], ymm5
        #vmovupd       [rip+IP], ymm12
		#INSTR    ymm6, [rip+PI]
		#INSTR    ymm7, [rip+PI]
		#INSTR    [rip+IP], ymm8
        vmovupd       [rip+PI], ymm13

		INSTR    ymm9, [rip+PI]
        #vmovupd      ymm11, [rip+PI]
		INSTR    ymm10, [rip+PI]
		#INSTR    [rip+IP], ymm11
        #vmovupd       [rip+IP], ymm14
		#INSTR    ymm12, [rip+PI]
		#INSTR    ymm13, [rip+PI]
		#INSTR    [rip+IP], ymm14
        vmovupd       [rip+PI], ymm15
		
        #INSTR    ymm15, [rip+PI]
		#INSTR    ymm16, [rip+PI]
        #INSTR    [rip+IP], ymm17
		#INSTR    ymm18, [rip+PI]
		#INSTR    ymm19, [rip+PI]
		#INSTR    [rip+IP], ymm20
		#INSTR    ymm21, [rip+PI]
		#INSTR    ymm22, [rip+PI]
		#INSTR    [rip+IP], ymm23
		#INSTR    ymm24, [rip+PI]
		#INSTR    ymm25, [rip+PI]
		#INSTR    [rip+IP], ymm26
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
