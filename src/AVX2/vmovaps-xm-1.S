#define INSTR vmovaps
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
        lea     rax, [rbp-32]
        mov     [rax], rax
loop:
		inc      i
		INSTR    xmm0, [rax]
        movq      rdx, xmm0
		INSTR    xmm0, [rdx]
        movq      r9, xmm0
		INSTR    xmm0, [r9]
        movq      r10, xmm0
        INSTR    xmm0, [r10]
        movq      r11, xmm0
        INSTR    xmm0, [r11]
        movq      r12, xmm0	
        INSTR    xmm0, [r12]
        movq      r13, xmm0	
        INSTR    xmm0, [r13]
        movq      r14, xmm0	
        INSTR    xmm0, [r14]
        movq      rax, xmm0	
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
