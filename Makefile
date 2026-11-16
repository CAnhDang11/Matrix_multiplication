CXX ?= g++
MPICXX ?= mpicxx

CXXFLAGS ?= -std=c++17 -O3 -fopenmp -Wall -Wextra
INCLUDES = -Iinclude

BIN_DIR = bin

SHARED_BINS = $(BIN_DIR)/naive_seq $(BIN_DIR)/naive_omp $(BIN_DIR)/strassen_seq $(BIN_DIR)/strassen_omp
MPI_BINS = $(BIN_DIR)/naive_mpi $(BIN_DIR)/naive_hybrid $(BIN_DIR)/strassen_mpi $(BIN_DIR)/strassen_hybrid

.PHONY: all shared mpi clean test

all: shared mpi

shared: $(SHARED_BINS)

mpi: $(MPI_BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Shared-memory targets
$(BIN_DIR)/naive_seq: src/common/common.cpp src/naive/naive_seq.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BIN_DIR)/naive_omp: src/common/common.cpp src/naive/naive_omp.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BIN_DIR)/strassen_seq: src/common/common.cpp src/strassen/strassen.cpp src/strassen/strassen_seq.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BIN_DIR)/strassen_omp: src/common/common.cpp src/strassen/strassen.cpp src/strassen/strassen_omp.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# MPI & Hybrid targets
$(BIN_DIR)/naive_mpi: src/common/common.cpp src/naive/naive_mpi.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BIN_DIR)/naive_hybrid: src/common/common.cpp src/naive/naive_hybrid.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BIN_DIR)/strassen_mpi: src/common/common.cpp src/strassen/strassen.cpp src/strassen/strassen_mpi.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BIN_DIR)/strassen_hybrid: src/common/common.cpp src/strassen/strassen.cpp src/strassen/strassen_hybrid.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

test: shared
	@echo "Chay kiem thu co ban..."
	@if [ -f tests/test_strassen.sh ]; then bash tests/test_strassen.sh; fi

clean:
	rm -rf $(BIN_DIR)
