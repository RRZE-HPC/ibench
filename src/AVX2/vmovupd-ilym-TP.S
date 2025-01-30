#define INSTR vmovupd
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
        push r12
        push r13
        push r14
        push r15
loop:
		inc      i
		INSTR    ymm3, [rip+PI]
		INSTR    ymm4, [rip+PI]
        mov      r9, [rip+PI]
		INSTR    ymm5, [rip+PI]
		INSTR    ymm6, [rip+PI]
        mov      r10, [rip+PI]
		INSTR    ymm7, [rip+PI]
		INSTR    ymm8, [rip+PI]
        mov      r11, [rip+PI]
		INSTR    ymm9, [rip+PI]
		INSTR    ymm10, [rip+PI]
        mov      r12, [rip+PI]
		INSTR    ymm11, [rip+PI]
		INSTR    ymm12, [rip+PI]
        mov      r13, [rip+PI]
		INSTR    ymm13, [rip+PI]
		INSTR    ymm14, [rip+PI]
        mov      r14, [rip+PI]
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
