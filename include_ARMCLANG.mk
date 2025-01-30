CC 		= 	armclang
AS 		= 	armclang
CFLAGS 	= 	-march=armv9-a+sve2 -O3 -DLIKWID_PERFMON ${LIKWID_INC}
ASFLAGS =   -march=armv9-a+sve2 -O3
#CFLAGS 	= 	-march=armv8.5-a -O3
#ASFLAGS =   -march=armv8.5-a -O3
# -msve-vector-bits=512 -march=armv8.2-a+sve
LFLAGS 	=  	-shared

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/BASE-ARM64/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/NEON/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SVE/*.S))
