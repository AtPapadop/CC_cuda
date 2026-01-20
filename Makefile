# Makefile — CUDA Connected Components test build
#
# Same as before, but installs the built binary into ./bin/

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CC        := gcc-12
CXX       := g++-12

TARGET    := cc_cuda_test

INCDIRS   := -Iinclude
SRCDIR    := src
BUILDDIR  := build
BINDIR    := bin

BIN       := $(BINDIR)/$(TARGET)

# ---- Matio (optional; used by your graph loader) ----
MATIO_INC ?=
MATIO_LIB ?=
MATIO_LDLIBS ?= -lmatio

# ---- Common flags ----
CFLAGS    := -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter $(INCDIRS)
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter $(INCDIRS)

# nvcc: compile host C++ in .cu; keep it simple/portable
NVCCFLAGS := -O3 -lineinfo $(INCDIRS) $(MATIO_INC) -ccbin=$(CXX) --compiler-options '$(CXXFLAGS)'

# Linker libs
LDLIBS    := $(MATIO_LIB) $(MATIO_LDLIBS) -lcudart

# ---- Sources ----
C_SRCS    := $(SRCDIR)/graph.c \
             $(SRCDIR)/opt_parser.c \
			 $(SRCDIR)/cc_cuda.c \
             $(SRCDIR)/results_writer.c

CU_SRCS   := $(SRCDIR)/cc_cuda_thread_per_vertex.cu \
			 $(SRCDIR)/cc_cuda_thread_per_vertex_frontier.cu \
			 $(SRCDIR)/cc_cuda_thread_per_vertex_afforest.cu \
			 $(SRCDIR)/cc_cuda_warp_per_vertex.cu \
			 $(SRCDIR)/cc_cuda_warp_per_vertex_frontier.cu \
			 $(SRCDIR)/cc_cuda_warp_per_vertex_afforest.cu \
			 $(SRCDIR)/cc_cuda_block_per_vertex.cu \
			 $(SRCDIR)/cc_cuda_block_per_vertex_frontier.cu \
			 $(SRCDIR)/cc_cuda_block_per_vertex_afforest.cu \
             $(SRCDIR)/cc_cuda_test.cu

C_OBJS    := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(C_SRCS))
CU_OBJS   := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(CU_SRCS))

OBJS      := $(C_OBJS) $(CU_OBJS)

.PHONY: all clean print

all: $(BIN)

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(BINDIR):
	@mkdir -p $(BINDIR)

# Compile C sources with gcc
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA sources with nvcc
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link with nvcc to pull in CUDA runtime properly
$(BIN): $(OBJS) | $(BINDIR)
	$(NVCC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

clean:
	@rm -rf $(BUILDDIR) $(BINDIR)

print:
	@echo "NVCC=$(NVCC)"
	@echo "CC=$(CC)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"
