CC 		= 	xlc
AS 		= 	xlc
CFLAGS 	= 	-O3
LFLAGS 	= 	-shared

KERNELS	= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/VSX/*.S))
