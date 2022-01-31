CC 		= 	gcc
AS 		= 	gcc
CFLAGS 	= 	-O3
# -msve-vector-bits=512 -march=armv8.2-a+sve
LFLAGS 	=  	-shared

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/NEON/*.S))
#KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SVE/*.S))
