CC 		= 	icc
AS 		= 	icc
CFLAGS 	= 	-O3 -mmic
LFLAGS 	= 	-shared -mmic

KERNELS	= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/gp/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/imci/*.S))
