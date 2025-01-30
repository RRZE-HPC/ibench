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
        # expand from AVX to AVX-512
        vinsertf64x4 zmm0, zmm0, ymm0, 0x1
        vpcmpeqq ymm2, ymm2, ymm2
        vmovapd zmm0, zmm0

        kxnorb k1, k1, k1
loop:
		inc      i
		INSTR    zmm0{k1}, [rsp + ymm2*8]
        vfmadd213pd zmm1, zmm1, zmm1
        vfmadd213pd zmm2, zmm2, zmm2
        vfmadd213pd zmm3, zmm3, zmm3
        vfmadd213pd zmm4, zmm4, zmm4
        vfmadd213pd zmm5, zmm5, zmm5
        vfmadd213pd zmm6, zmm6, zmm6
        vfmadd213pd zmm7, zmm7, zmm7
        #vfmadd213pd zmm8, zmm8, zmm8
        #vfmadd213pd zmm9, zmm9, zmm9
        #vfmadd213pd zmm10, zmm10, zmm10
        #vfmadd213pd zmm11, zmm11, zmm11
        #vfmadd213pd zmm12, zmm12, zmm12
		INSTR    zmm0{k1}, [rsp + ymm2*8]
        vfmadd213pd zmm13, zmm13, zmm13
        vfmadd213pd zmm14, zmm14, zmm14
        vfmadd213pd zmm15, zmm15, zmm15
        vfmadd213pd zmm16, zmm16, zmm16
        vfmadd213pd zmm17, zmm17, zmm17
        vfmadd213pd zmm18, zmm18, zmm18
        vfmadd213pd zmm19, zmm19, zmm19
        #vfmadd213pd zmm20, zmm20, zmm20
        #vfmadd213pd zmm21, zmm21, zmm21
        #vfmadd213pd zmm22, zmm22, zmm22
        #vfmadd213pd zmm23, zmm23, zmm23
        #vfmadd213pd zmm24, zmm24, zmm24
		INSTR    zmm0{k1}, [rsp + ymm2*8]
        #vfmadd213pd zmm1, zmm1, zmm1
        #vfmadd213pd zmm2, zmm2, zmm2
        #vfmadd213pd zmm3, zmm3, zmm3
        #vfmadd213pd zmm4, zmm4, zmm4
        #vfmadd213pd zmm5, zmm5, zmm5
        vfmadd213pd zmm6, zmm6, zmm6
        vfmadd213pd zmm7, zmm7, zmm7
        vfmadd213pd zmm8, zmm8, zmm8
        vfmadd213pd zmm9, zmm9, zmm9
        vfmadd213pd zmm10, zmm10, zmm10
        vfmadd213pd zmm11, zmm11, zmm11
        vfmadd213pd zmm12, zmm12, zmm12
		INSTR    zmm0{k1}, [rsp + ymm2*8]
        vfmadd213pd zmm13, zmm13, zmm13
        vfmadd213pd zmm14, zmm14, zmm14
        #vfmadd213pd zmm15, zmm15, zmm15
        #vfmadd213pd zmm16, zmm16, zmm16
        #vfmadd213pd zmm17, zmm17, zmm17
        #vfmadd213pd zmm18, zmm18, zmm18
        #vfmadd213pd zmm19, zmm19, zmm19
        vfmadd213pd zmm20, zmm20, zmm20
        vfmadd213pd zmm21, zmm21, zmm21
        vfmadd213pd zmm22, zmm22, zmm22
        vfmadd213pd zmm23, zmm23, zmm23
        vfmadd213pd zmm24, zmm24, zmm24
		INSTR    zmm0{k1}, [rsp + ymm2*8]
        vfmadd213pd zmm1, zmm1, zmm1
        vfmadd213pd zmm2, zmm2, zmm2
        vfmadd213pd zmm3, zmm3, zmm3
        vfmadd213pd zmm4, zmm4, zmm4
        vfmadd213pd zmm5, zmm5, zmm5
        vfmadd213pd zmm6, zmm6, zmm6
        vfmadd213pd zmm7, zmm7, zmm7
        #vfmadd213pd zmm8, zmm8, zmm8
        #vfmadd213pd zmm9, zmm9, zmm9
        #vfmadd213pd zmm10, zmm10, zmm10
        #vfmadd213pd zmm11, zmm11, zmm11
        #vfmadd213pd zmm12, zmm12, zmm12
		INSTR    zmm0{k1}, [rsp + ymm2*8]
        vfmadd213pd zmm13, zmm13, zmm13
        vfmadd213pd zmm14, zmm14, zmm14
        vfmadd213pd zmm15, zmm15, zmm15
        vfmadd213pd zmm16, zmm16, zmm16
        vfmadd213pd zmm17, zmm17, zmm17
        vfmadd213pd zmm18, zmm18, zmm18
        vfmadd213pd zmm19, zmm19, zmm19
        #vfmadd213pd zmm20, zmm20, zmm20
        #vfmadd213pd zmm21, zmm21, zmm21
        #vfmadd213pd zmm22, zmm22, zmm22
        #vfmadd213pd zmm23, zmm23, zmm23
        #vfmadd213pd zmm24, zmm24, zmm24
		cmp      i, N
		jl       loop
done:
		mov	rsp, rbp
		pop	rbp
		ret
.size latency, .-latency
