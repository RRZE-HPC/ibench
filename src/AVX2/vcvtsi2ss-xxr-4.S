#define INSTR vcvtsi2ss
#define FPIN vfmadd213pd
#define EXIN add
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
		# copy DP 1.0
		vmovaps xmm0, xmm0
		vmovaps xmm1, xmm0
        vmovaps xmm8, xmm0
        vmovaps xmm9, xmm0
        vmovaps xmm10, xmm0
        vmovaps xmm11, xmm0
        vmovaps xmm12, xmm0
        vmovaps xmm13, xmm0
        vmovaps xmm14, xmm0
        vmovaps xmm15, xmm0
		# Create DP 2.0
		vaddpd xmm1, xmm1, xmm1
        vaddpd xmm12, xmm12, xmm12
		# Create DP 0.5
		vdivpd xmm2, xmm0, xmm1
loop:
		inc      i
		INSTR    xmm0, xmm1, eax
#        FPIN xmm10, xmm11, xmm12
        EXIN     r10, r11
		INSTR    xmm2, xmm3, eax
#        FPIN xmm13, xmm14, xmm12
        EXIN     r13, r14
		INSTR    xmm4, xmm5, eax
#        FPIN xmm15, xmm12, xmm12
        EXIN     r15, r12
		INSTR    xmm6, xmm7, eax
#        FPIN xmm8, xmm9, xmm12
        EXIN     rdx, r9
        INSTR    xmm1, xmm0, eax
#        FPIN xmm11, xmm10, xmm12
        EXIN     r11, r10
		INSTR    xmm3, xmm2, eax
#        FPIN xmm14, xmm13, xmm12
        EXIN     r14, r13
		INSTR    xmm5, xmm4, eax
#        FPIN xmm12, xmm15, xmm12
        EXIN     r12, r15
		INSTR    xmm7, xmm6, eax
#        FPIN xmm9, xmm8, xmm12
        EXIN     r9, rdx
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
