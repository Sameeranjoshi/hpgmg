# find MPI compiler
CC=`which mpicc`
#CC=`which mpiicc`

# find NVCC compiler
NVCC=`which nvcc`

# set gpu architectures to compile for
#CUDA_ARCH+="-gencode arch=compute_60,code=sm_60 "
#CUDA_ARCH+="-gencode arch=compute_70,code=sm_70 "
#CUDA_ARCH+="-gencode arch=compute_80,code=sm_80 "
CUDA_ARCH+="-gencode arch=compute_90,code=sm_90 "
FOLDER="H200_fp32"

# main tile size
OPTS+="-DBLOCKCOPY_TILE_I=32 "
OPTS+="-DBLOCKCOPY_TILE_J=4 "
OPTS+="-DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS+="-DBOUNDARY_TILE_I=64 "
OPTS+="-DBOUNDARY_TILE_J=16 "
OPTS+="-DBOUNDARY_TILE_K=16 "

# host level threshold: number of grid elements
OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=0 "

# max number of solves after warmup
OPTS+="-DMAX_SOLVES=10 "

# unified memory allocation options
OPTS+="-DCUDA_UM_ALLOC "
OPTS+="-DCUDA_UM_ZERO_COPY "

# MPI buffers allocation policy
OPTS+="-DMPI_ALLOC_ZERO_COPY "
#OPTS+="-DMPI_ALLOC_PINNED "

# stencil optimizations
OPTS+="-DUSE_REG "
OPTS+="-DUSE_TEX "
#OPTS+="-DUSE_SHM "

# GSRB smoother options
#OPTS+="-DGSRB_FP "
#OPTS+="-DGSRB_STRIDE2 "
#OPTS+="-DGSRB_BRANCH "
#OPTS+="-DGSRB_OOP "

# tools
#OPTS+="-DUSE_PROFILE "
#OPTS+="-DUSE_NVTX "
#OPTS+="-DUSE_ERROR "

# override MVAPICH flags for C++
OPTS+="-DMPICH_IGNORE_CXX_SEEK "
OPTS+="-DMPICH_SKIP_MPICXX "

# OPTS+="-DUSE_PERIODIC_BC "
OPTS+="-DUSE_DIRICHLET_BC "

rm -rf $FOLDER
# export MPICH_GPU_SUPPORT_ENABLED=1
# export CRAY_ACCEL_TARGET=nvidia80
# export LDFLAGS=-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/ 
# export LD_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.28/gtl/lib/:$LD_LIBRARY_PATH
LDLIBS+="-ldl"

# GSRB smoother (default)
python ./configure --arch="$FOLDER" --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo $OPTS" --CUDAARCH="$CUDA_ARCH" --LDLIBS="$LDLIBS" --no-fe --no-fv-mpi --fv-cycle="V" --fv-smoother="jacobi"

# Chebyshev smoother
# ./configure --CC=$CC --NVCC=$NVCC --CFLAGS="-O1 -fopenmp $OPTS" --NVCCFLAGS="-O1 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --fv-smoother="cheby" --no-fe

#make clean -C build
make -j123 -C $FOLDER V=1
