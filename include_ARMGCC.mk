CC 		= 	gcc
AS 		= 	gcc
CFLAGS 	= 	-O3
LFLAGS 	=  	-shared

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/NEON/*.S))
#KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SVE/*.S))
