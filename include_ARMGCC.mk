CC 		= 	gcc
AS 		= 	gcc
CFLAGS 	= 	-O3 -march=armv8.5-a
# -msve-vector-bits=512 -march=armv8.2-a+sve
ASFLAGS =   -O3 -march=armv8.5-a
LFLAGS 	=  	-shared

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/BASE-ARM64/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/NEON/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SVE/*.S))
