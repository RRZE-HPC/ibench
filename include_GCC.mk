CC 		= 	gcc
AS 		= 	gcc
#CFLAGS 	= 	-O3 -march=rv64gcv -x assembler-with-cpp
CFLAGS 	= 	-O3 -march=rv64gcv
#ASFLAGS =   -O3 -march=rv64gcv -x assembler-with-cpp
ASFLAGS =   -O3 -march=rv64gcv
LFLAGS 	= 	-shared

KERNELS    +=  $(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/BASE-RISC-V/*.S))
KERNELS    +=  $(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/RVV/*.S))
#KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/BASE-X86/*.S))
#KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/SSE4.2/*.S))
#KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX2/*.S))
#KERNELS	+= 	$(patsubst $(SRC_DIR)/%.S, %.so, $(wildcard $(SRC_DIR)/AVX512/*.S))
