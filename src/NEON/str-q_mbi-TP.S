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
        INSTR    q2,  [sp, x24] 
        INSTR    q6,  [sp, x25]  
        INSTR    q8,  [sp, x28]  
        INSTR    q10, [sp, x27]  
        INSTR    q12, [sp, x24]  
        INSTR    q14, [sp, x25]  
        INSTR    q16, [sp, x28]  
        INSTR    q18, [sp, x27]  
        INSTR    q2,  [sp, x24] 
        INSTR    q6,  [sp, x25]  
        INSTR    q8,  [sp, x28]  
        INSTR    q10, [sp, x27]  
        INSTR    q12, [sp, x24]  
        INSTR    q14, [sp, x25]  
        INSTR    q16, [sp, x28]  
        INSTR    q18, [sp, x27]  
        INSTR    q2,  [sp, x24]
        INSTR    q6,  [sp, x25]        
        INSTR    q8,  [sp, x28]        
        INSTR    q10, [sp, x27]        
        INSTR    q12, [sp, x24]        
        INSTR    q14, [sp, x25]        
        INSTR    q16, [sp, x28]        
        INSTR    q18, [sp, x27]
        INSTR    q2,  [sp, x24]
        INSTR    q6,  [sp, x25]        
        INSTR    q8,  [sp, x28]        
        INSTR    q10, [sp, x27]        
        INSTR    q12, [sp, x24]        
        INSTR    q14, [sp, x25]        
        INSTR    q16, [sp, x28]        
        INSTR    q18, [sp, x27]        
        INSTR    q2,  [sp, x24] 
        INSTR    q6,  [sp, x25]  
        INSTR    q8,  [sp, x28]  
        INSTR    q10, [sp, x27]  
        INSTR    q12, [sp, x24]  
        INSTR    q14, [sp, x25]  
        INSTR    q16, [sp, x28]  
        INSTR    q18, [sp, x27]  
        INSTR    q2,  [sp, x24] 
        INSTR    q6,  [sp, x25]  
        INSTR    q8,  [sp, x28]  
        INSTR    q10, [sp, x27]  
        INSTR    q12, [sp, x24]  
        INSTR    q14, [sp, x25]  
        INSTR    q16, [sp, x28]  
        INSTR    q18, [sp, x27]  
        INSTR    q2,  [sp, x24]
        INSTR    q6,  [sp, x25]        
        INSTR    q8,  [sp, x28]        
        INSTR    q10, [sp, x27]        
        INSTR    q12, [sp, x24]        
        INSTR    q14, [sp, x25]        
        INSTR    q16, [sp, x28]        
        INSTR    q18, [sp, x27]
        INSTR    q2,  [sp, x24]
        INSTR    q6,  [sp, x25]        
        INSTR    q8,  [sp, x28]        
        INSTR    q10, [sp, x27]        
        INSTR    q12, [sp, x24]        
        INSTR    q14, [sp, x25]        
        INSTR    q16, [sp, x28]        
        INSTR    q18, [sp, x27]        
        bne       loop
done:

        # pop callee-save registers from stack
        ld1            {v12.2d, v13.2d, v14.2d, v15.2d}, [sp]
        add            sp, sp, #64
        ld1            {v8.2d, v9.2d, v10.2d, v11.2d}, [sp]
        add            sp, sp, #64
        
        ret

.size latency, .-latency
