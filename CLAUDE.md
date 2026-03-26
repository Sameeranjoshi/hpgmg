# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HPGMG (High-Performance Geometric Multigrid) — a GPU-accelerated finite-volume Poisson solver using CUDA with Unified Memory. One MPI rank per GPU. Originally from LBNL/NVIDIA.

## Build Commands

```bash
# Recommended
module load cuda/12.5.3 openmpi python
# change build.sh for any tuning params
./build.sh
# Edit: Specify the build folder name generated from above script
./run.sh
_______________________________________________________________________________________________________
# Manual configure + build
python ./configure --arch=BUILDDIR --CC=$(which mpicc) --NVCC=$(which nvcc) \
  --CFLAGS="-O2 -fopenmp" --NVCCFLAGS="-O2" \
  --CUDAARCH="-gencode arch=compute_90,code=sm_90" \
  --fv-smoother=jacobi --fv-coarse-solver=cg --fv-cycle=V \
  --no-fe --no-fv-mpi
make -j -C BUILDDIR

# Run (args: log2_grid_size num_warmup_solves)
./BUILDDIR/bin/hpgmg-fv 8 1          # 256^3 grid, 1 warmup
mpirun -np 4 ./BUILDDIR/bin/hpgmg-fv 8 1   # multi-GPU
```

There are no automated tests — validation is by checking solver convergence and comparing residual norms in output.

## Architecture

**Build system:** `configure` (Python, via `hpgmgconf.py`) generates a per-arch Makefile. `base.mk` has the actual build rules using recursive `local.mk` files. Build artifacts go in the arch-named directory (e.g., `H100build_fp64/`).

**Solver pipeline (finite-volume/source/):**

- `hpgmg-fv.c` — entry point, benchmark harness, GPU device setup
- `mg.c` — multigrid algorithm: FMG solver, V/F/U-cycles, MGPCG
- `level.c` — grid level creation, memory allocation, box decomposition
- `operators.*.c` — CPU operator implementations (apply_op, residual, smooth, restrict, interpolate)
- `solvers.c` + `solvers/*.c` — bottom solvers (CG, BiCGStab, CA variants)
- `defines.h` — vector slot IDs used throughout

**GPU kernels (finite-volume/source/cuda/):**

- `operators.fv2.cu` / `operators.fv4.cu` — GPU operator implementations
- `cuda/stencils/*.h` — optimized stencil variants (register, texture, shared memory)
- `cuda/blockCopy.h` — block copy operations for ghost exchange, restriction, interpolation
- `cuda/common.h` — CUDA error checking, memory management wrappers

**Key data structures (level.h):**

- `level_type` — grid level: dimensions, box decomposition, vector storage, communicators, timers
- `box_type` — local box metadata and vector pointers
- `blockCopy_type` — encodes data movement operations between grids/buffers
- `mg_type` — multigrid hierarchy (array of level_type pointers)

## Key Configuration Options (build.sh)

Controlled via `-D` flags in OPTS:

- `HOST_LEVEL_SIZE_THRESHOLD=N` — grid elements below which levels run on CPU (0 = all GPU)
- `BLOCKCOPY_TILE_{I,J,K}` / `BOUNDARY_TILE_{I,J,K}` — CUDA kernel tile sizes
- `USE_REG`, `USE_TEX`, `USE_SHM` — stencil optimization strategies
- `CUDA_UM_ALLOC`, `CUDA_UM_ZERO_COPY` — unified memory policies
- `USE_DIRICHLET_BC` or `USE_PERIODIC_BC` — boundary condition type
- `MAX_SOLVES` — number of timed benchmark solves

Configure script selects: `--fv-smoother` (jacobi/gsrb/cheby/l1jacobi), `--fv-coarse-solver` (cg/bicgstab/cacg/cabicgstab), `--fv-cycle` (V/F/U).

## GPU Architecture Targets

Set `CUDA_ARCH` in build.sh: compute_60 (P100), compute_70 (V100), compute_80 (A100), compute_90 (H100).