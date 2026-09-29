CC ?= clang
CFLAGS ?= -O2 -Wall -Wextra -std=c11
BUILD := build

all: $(BUILD)/latency $(BUILD)/bandwidth

$(BUILD)/%: %.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< -lpthread

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
