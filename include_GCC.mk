CC 		= 	gcc
AS 		= 	gcc
CFLAGS 	= 	-O3
ASFLAGS =   -O3
LFLAGS 	= 	-z noexecstack -shared

KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/BASE-X86/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SSE4.2/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX2/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX-512/*.S))
KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX-512-FP16/*.S))
