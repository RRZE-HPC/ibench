CC 		= 	gcc
AS 		= 	gcc
CFLAGS 	= 	-O3 -x assembler-with-cpp
ASFLAGS =   -O3 -x assembler-with-cpp
LFLAGS 	= 	-shared

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/BASE-X86/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SSE4.2/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX2/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX512/*.S))
