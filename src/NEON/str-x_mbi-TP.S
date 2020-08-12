#define INSTR str
#define NINST 64
#define N x0

.globl ninst
.data
ninst:
.long NINST
.text
.globl latency
.type latency, @function
.align 2
latency:

        # push callee-save registers onto stack
        sub            sp, sp, #64
        st1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        sub            sp, sp, #64
        st1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        stp x29, x30, [sp, -96]!
        stp x19, x20, [sp, 16]
        stp x21, x22, [sp, 32]
        stp x24, x25, [sp, 48]
        stp x26, x27, [sp, 64]
        str x28, [sp, 80]


        mov     x4, N

        fmov    v0.2d, #1.00000000
        fmov    v1.2d, #1.00000000
        fmov    v2.2d, #1.00000000
        mov     x24, #-64
        mov     x25, #-128
        mov     x28, #-192
        mov     x27, #-256

loop:
        subs      x4, x4, #1
        INSTR    x2,  [sp, x24] 
        INSTR    x6,  [sp, x25]  
        INSTR    x8,  [sp, x28]  
        INSTR    x10, [sp, x27]  
        INSTR    x12, [sp, x24]  
        INSTR    x14, [sp, x25]  
        INSTR    x16, [sp, x28]  
        INSTR    x18, [sp, x27]  
        INSTR    x2,  [sp, x24] 
        INSTR    x6,  [sp, x25]  
        INSTR    x8,  [sp, x28]  
        INSTR    x10, [sp, x27]  
        INSTR    x12, [sp, x24]  
        INSTR    x14, [sp, x25]  
        INSTR    x16, [sp, x28]  
        INSTR    x18, [sp, x27]  
        INSTR    x2,  [sp, x24]
        INSTR    x6,  [sp, x25]        
        INSTR    x8,  [sp, x28]        
        INSTR    x10, [sp, x27]        
        INSTR    x12, [sp, x24]        
        INSTR    x14, [sp, x25]        
        INSTR    x16, [sp, x28]        
        INSTR    x18, [sp, x27]
        INSTR    x2,  [sp, x24]
        INSTR    x6,  [sp, x25]        
        INSTR    x8,  [sp, x28]        
        INSTR    x10, [sp, x27]        
        INSTR    x12, [sp, x24]        
        INSTR    x14, [sp, x25]        
        INSTR    x16, [sp, x28]        
        INSTR    x18, [sp, x27]        
        INSTR    x2,  [sp, x24] 
        INSTR    x6,  [sp, x25]  
        INSTR    x8,  [sp, x28]  
        INSTR    x10, [sp, x27]  
        INSTR    x12, [sp, x24]  
        INSTR    x14, [sp, x25]  
        INSTR    x16, [sp, x28]  
        INSTR    x18, [sp, x27]  
        INSTR    x2,  [sp, x24] 
        INSTR    x6,  [sp, x25]  
        INSTR    x8,  [sp, x28]  
        INSTR    x10, [sp, x27]  
        INSTR    x12, [sp, x24]  
        INSTR    x14, [sp, x25]  
        INSTR    x16, [sp, x28]  
        INSTR    x18, [sp, x27]  
        INSTR    x2,  [sp, x24]
        INSTR    x6,  [sp, x25]        
        INSTR    x8,  [sp, x28]        
        INSTR    x10, [sp, x27]        
        INSTR    x12, [sp, x24]        
        INSTR    x14, [sp, x25]        
        INSTR    x16, [sp, x28]        
        INSTR    x18, [sp, x27]
        INSTR    x2,  [sp, x24]
        INSTR    x6,  [sp, x25]        
        INSTR    x8,  [sp, x28]        
        INSTR    x10, [sp, x27]        
        INSTR    x12, [sp, x24]        
        INSTR    x14, [sp, x25]        
        INSTR    x16, [sp, x28]        
        INSTR    x18, [sp, x27]        
        bne       loop
done:

        # pop callee-save registers from stack
        ldp x19, x20, [sp, 16]
        ldp x21, x22, [sp, 32]
        ldp x24, x25, [sp, 48]
        ldp x26, x27, [sp, 64]
        ldr x28, [sp, 80]
        ldp x29, x30, [sp], 96
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
