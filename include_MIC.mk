CC 		= 	icc
AS 		= 	icc
CFLAGS 	= 	-O3 -mmic
LFLAGS 	= 	-shared -mmic

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/IMCI/*.S))
