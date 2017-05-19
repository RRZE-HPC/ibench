CC 		= 	icc
AS 		= 	icc
CFLAGS 	= 	-O3
LFLAGS 	=  	-shared

KERNELS	= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/gp/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/sse/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX-512/*.S))
