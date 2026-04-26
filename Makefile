# Makefile for the C++ inference engine.
#
# The .cpp picks Accelerate (macOS, via __APPLE__) or OpenBLAS (Linux, when
# -DUSE_OPENBLAS is set) automatically; on Linux without libopenblas-dev the
# blas targets won't build, so they're opt-in.
#
#   make           # naive, sparse, and their *_prof variants (no extra deps)
#   make blas      # blas + blas_prof (Linux: needs libopenblas-dev; macOS: ok)
#   make profile   # the three *_prof variants only
#   make all       # everything (= make + make blas)
#   make clean     # remove build/

CXX      ?= g++
SRC      := transformer_vm/model/transformer.cpp
ATTN     := transformer_vm/attention
BUILD    := build
COMMON   := -std=c++17 -O3 -march=native -I $(ATTN)
PROF     := -DPROFILE_PHASES

UNAME_S  := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  BLAS_FLAGS := -framework Accelerate
else
  BLAS_FLAGS := -DUSE_OPENBLAS -lopenblas
endif

BINS_DEFAULT := \
    $(BUILD)/transformer_naive  $(BUILD)/transformer_sparse \
    $(BUILD)/transformer_naive_prof  $(BUILD)/transformer_sparse_prof
BINS_BLAS    := $(BUILD)/transformer_blas $(BUILD)/transformer_blas_prof
BINS_PROF    := $(BUILD)/transformer_naive_prof $(BUILD)/transformer_sparse_prof

.PHONY: default all blas profile clean
default: $(BINS_DEFAULT)
blas:    $(BINS_BLAS)
profile: $(BINS_PROF)
all:     default blas

$(BUILD):
	@mkdir -p $@

# On Linux, libraries must follow their users on the link line for
# --as-needed resolution. macOS is order-insensitive.
ifeq ($(UNAME_S),Darwin)
  BLAS_DEFS := -DUSE_OPENBLAS
  BLAS_LIBS := $(BLAS_FLAGS)
else
  BLAS_DEFS := -DUSE_OPENBLAS
  BLAS_LIBS := -lopenblas
endif

$(BUILD)/transformer_naive: $(SRC) | $(BUILD)
	$(CXX) $(COMMON) $< -o $@

$(BUILD)/transformer_blas: $(SRC) | $(BUILD)
	$(CXX) $(COMMON) $(BLAS_DEFS) $< -o $@ $(BLAS_LIBS)

$(BUILD)/transformer_sparse: $(SRC) | $(BUILD)
	$(CXX) $(COMMON) -DUSE_SPARSE_PROJ $< -o $@

$(BUILD)/transformer_naive_prof: $(SRC) | $(BUILD)
	$(CXX) $(COMMON) $(PROF) $< -o $@

$(BUILD)/transformer_blas_prof: $(SRC) | $(BUILD)
	$(CXX) $(COMMON) $(PROF) $(BLAS_DEFS) $< -o $@ $(BLAS_LIBS)

$(BUILD)/transformer_sparse_prof: $(SRC) | $(BUILD)
	$(CXX) $(COMMON) $(PROF) -DUSE_SPARSE_PROJ $< -o $@

clean:
	rm -rf $(BUILD)
