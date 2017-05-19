CC 		= 	gcc
AS 		= 	gcc
CFLAGS 	= 	-O3 -x assembler-with-cpp
LFLAGS 	= 	-shared

KERNELS	= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/gp/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/sse/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/avx/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/avx2/*.S))
