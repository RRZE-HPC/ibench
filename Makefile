COMPILER=NEON

TARGET 	= 	ibench
SRC_DIR = 	src
KDIRS	+= 	$(patsubst $(SRC_DIR)/%, %, $(wildcard $(SRC_DIR)/*))
Q 		= 	@

include include_$(COMPILER).mk

$(TARGET): ibench.c $(KDIRS) $(KERNELS)
	$(Q)echo "===>  COMPILING $@"
	$(Q)$(CC) $(CFLAGS) $< -o $@ -ldl 

$(KDIRS):
	$(Q)mkdir $(KDIRS)

%.so:
	$(Q)echo "===>  ASSEMBLING $@"
	$(Q)$(AS) $(LFLAGS) $(patsubst %.so, $(SRC_DIR)/%.S, $@) -o $@

.PHONY: clean

clean:
	$(Q)echo "===>  CLEAN"
	$(Q)rm -rf $(KDIRS)
	$(Q)rm -f $(TARGET)
