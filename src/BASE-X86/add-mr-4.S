#define INSTR add
#define NINST 8
#define N edi
#define i r8d


.intel_syntax noprefix
.globl ninst
.data
ninst:
.long NINST
.align 32
PI:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI2:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI3:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI4:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI5:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI6:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI7:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9
PI8:
.long  0xf01b866e, 0x400921f9, 0xf01b866e, 0x400921f9

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
		push    rax
		push    rbx
		push    rcx
		push    rdx
		push    r9
		push    r10
		push    r11
		push    r12
		push    r13
		push    r14
		push    r15
		xor     rax, rax
		xor     rbx, rbx
		xor     rcx, rcx
		xor     rdx, rdx
		xor     r9, r9
		xor     r10, r10
		xor     r11, r11
		xor     r12, r12
		xor     r13, r13
		xor     r14, r14
		xor     r15, r15
#        lea     rax, [rip+PI]
#        mov     [rax], rax
#        lea     rbx, [rip+PI]
#        mov     [rbx], rbx
#        lea     rcx, [rip+PI]
#        mov     [rcx], rcx
#        lea     rdx, [rip+PI]
#        mov     [rdx], rdx
loop:
		inc      i
		INSTR    [rip+PI], r10 
		INSTR    [rip+PI2], r11 
		INSTR    [rip+PI3], r12 
		INSTR    [rip+PI4], r13 
		INSTR    [rip+PI5], rax 
		INSTR    [rip+PI6], rbx 
		INSTR    [rip+PI7], rcx 
		INSTR    [rip+PI8], rdx 
		cmp      i, N
		jl       loop
		pop     r15
		pop     r14
		pop     r13
		pop     r12
		pop     r11
		pop     r10
		pop     r9
		pop     rdx
		pop     rcx
		pop     rbx
		pop     rax
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
