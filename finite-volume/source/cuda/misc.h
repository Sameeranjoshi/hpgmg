/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define MISC_THREAD_BLOCK_SIZE		256

// Include defines.h for REAL type and macros
#include "../defines.h"

// Define CUDA type conversion macros based on REAL type
#ifdef USE_FLOAT32
  // For float: use int (32-bit) for atomic operations
  #define __REAL_as_int(x) __float_as_int(x)
  #define __int_as_REAL(x) __int_as_float(x)
  typedef unsigned int REAL_int_type;
#else
  // For double: use long long (64-bit) for atomic operations
  #define __REAL_as_int(x) __double_as_longlong(x)
  #define __int_as_REAL(x) __longlong_as_double(x)
  typedef unsigned long long int REAL_int_type;
#endif

// CUB library is used for reductions
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include "extern/cub/cub.cuh"
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ REAL atomicAdd(REAL* address, REAL val)
{
    REAL_int_type* address_as_int = (REAL_int_type*)address;
    REAL_int_type old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __REAL_as_int(val +
                        __int_as_REAL(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __int_as_REAL(old);
}
#endif

__device__ REAL atomicMax(REAL* address, REAL val)
{
    REAL_int_type* address_as_int = (REAL_int_type*)address;
    REAL_int_type old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __REAL_as_int(max(val,
                        __int_as_REAL(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __int_as_REAL(old);
}

// this kernel zeros out the grid
__global__ void zero_vector_kernel(level_type level, int component_id)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    const int     dim = level.my_boxes[box].dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts;
    if(jlo<=  0)jlo-=ghosts;
    if(klo<=  0)klo-=ghosts;
    if(ihi>=dim)ihi+=ghosts;
    if(jhi>=dim)jhi+=ghosts;
    if(khi>=dim)khi+=ghosts;

    REAL * __restrict__ grid = level.my_boxes[box].vectors[component_id] + ghosts*(1+jStride+kStride);

    // note here that (ihi - ilo) can be greater than dim becasue of ghost blocks
    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i >= ihi) return;

    int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;

    for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
      for (int k = klo; k < khi; k++) {
        const int ijk = i + j*jStride + k*kStride;
        grid[ijk] = 0.0;
      }
}

// this kernel provides a generic axpy/mul implementation:
// if mul_vectors = 1: c = scale_a * a * b
// if mul_vectors = 0: c = scale_a * a + scale_b * b + shift_a
template <int mul_vectors>
__global__ void axpy_vector_kernel(level_type level, int id_c, REAL scale_a, REAL shift_a, REAL scale_b, int id_a, int id_b)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    REAL * __restrict__ grid_c = level.my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    REAL * __restrict__ grid_a = level.my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);
    REAL * __restrict__ grid_b = level.my_boxes[box].vectors[id_b] + ghosts*(1+jStride+kStride);

    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i >= ihi) return;

    int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;

    for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
      for (int k = klo; k < khi; k++) {
        const int ijk = i + j*jStride + k*kStride;
        if (mul_vectors)
          grid_c[ijk] = scale_a*grid_a[ijk]*grid_b[ijk];
        else
          grid_c[ijk] = scale_a*grid_a[ijk] + scale_b*grid_b[ijk] + shift_a;
      }
}

// simple coloring kernel, see misc.c for details
__global__ void color_vector_kernel(level_type level, int id_a, int colors_in_each_dim, int icolor, int jcolor, int kcolor)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int boxlowi = level.my_boxes[box].low.i;
    const int boxlowj = level.my_boxes[box].low.j;
    const int boxlowk = level.my_boxes[box].low.k;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    REAL * __restrict__ grid = level.my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i >= ihi) return;

    int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;

    for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
      for (int k = klo; k < khi; k++) {
        REAL sk=0.0;if( ((k+boxlowk+kcolor)%colors_in_each_dim) == 0 )sk=1.0; // if colors_in_each_dim==1 (don't color), all cells are set to 1.0
        REAL sj=0.0;if( ((j+boxlowj+jcolor)%colors_in_each_dim) == 0 )sj=1.0;
        REAL si=0.0;if( ((i+boxlowi+icolor)%colors_in_each_dim) == 0 )si=1.0;
        const int ijk = i + j*jStride + k*kStride;
        grid[ijk] = si*sj*sk;
      }
}

// 0: summation, 1: maximum absolute
template <int red_type>
__global__ void reduction_kernel(level_type level, int id, REAL *res)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    REAL * __restrict__ grid = level.my_boxes[box].vectors[id] + ghosts*(1+jStride+kStride);

    // accumulate per thread first (multiple elements)
    REAL thread_val = 0.0;

    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i < ihi) {
      int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;
      for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
        for (int k = klo; k < khi; k++) {
          const int ijk = i + j*jStride + k*kStride;
          REAL val = grid[ijk];
          switch (red_type) {
          case 0: thread_val += val; break;
          case 1: thread_val = max(thread_val, fabs(val)); break;
          }
        }
     }

  typedef cub::BlockReduce<REAL, MISC_THREAD_BLOCK_SIZE> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  REAL block_val;
  switch (red_type) {
  case 0:
    block_val = BlockReduceT(temp_storage).Sum(thread_val);
    if (threadIdx.x == 0) atomicAdd(res, block_val);
    break;
  case 1:
    block_val = BlockReduceT(temp_storage).Reduce(thread_val, cub::Max());
    if (threadIdx.x == 0) atomicMax(res, block_val);
    break;
  }
}

extern "C"
void cuda_zero_vector(level_type d_level, int id)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  zero_vector_kernel<<<grid, block>>>(d_level, id);
  CUDA_ERROR
}

extern "C"
void cuda_scale_vector(level_type d_level, int id_c, REAL scale_a, int id_a)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<0><<<grid, block>>>(d_level, id_c, scale_a, 0.0, 0.0, id_a, id_a);
  CUDA_ERROR
}

extern "C"
void cuda_shift_vector(level_type d_level, int id_c, REAL shift_a, int id_a)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<0><<<grid, block>>>(d_level, id_c, 1.0, shift_a, 0.0, id_a, id_a);
  CUDA_ERROR
}

extern "C"
void cuda_mul_vectors(level_type d_level, int id_c, REAL scale, int id_a, int id_b)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<1><<<grid, block>>>(d_level, id_c, scale, 0.0, 0.0, id_a, id_b);
  CUDA_ERROR
}

extern "C"
void cuda_add_vectors(level_type d_level, int id_c, REAL scale_a, int id_a, REAL scale_b, int id_b)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<0><<<grid, block>>>(d_level, id_c, scale_a, 0.0, scale_b, id_a, id_b);
  CUDA_ERROR
}

extern "C"
REAL cuda_sum(level_type d_level, int id)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return 0.0;

  REAL *d_res;
  REAL h_res[1];
  CUCHK( cudaMallocManaged((void**)&d_res, sizeof(REAL), cudaMemAttachGlobal) )
  CUCHK( cudaMemsetAsync(d_res, 0, sizeof(REAL)) )

  reduction_kernel<0><<<grid, block>>>(d_level, id, d_res);
  CUDA_ERROR

  // sync here to guarantee that the result is updated on GPU
  CUCHK( cudaDeviceSynchronize() )
  h_res[0] = d_res[0];

  CUCHK( cudaFree(d_res) )
  return h_res[0];
}

extern "C"
REAL cuda_max_abs(level_type d_level, int id)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return 0.0;

  REAL *d_res;
  REAL h_res[1];
  CUCHK( cudaMallocManaged((void**)&d_res, sizeof(REAL), cudaMemAttachGlobal) )
  CUCHK( cudaMemsetAsync(d_res, 0, sizeof(REAL)) )

  reduction_kernel<1><<<grid, block>>>(d_level, id, d_res);
  CUDA_ERROR

  // sync here to guarantee that the result is updated on GPU
  CUCHK( cudaDeviceSynchronize() )
  h_res[0] = d_res[0];

  CUCHK( cudaFree(d_res) )
  return h_res[0];
}

extern "C"
void cuda_color_vector(level_type d_level, int id_a, int colors_in_each_dim, int icolor, int jcolor, int kcolor)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  color_vector_kernel<<<grid, block>>>(d_level, id_a, colors_in_each_dim, icolor, jcolor, kcolor);
  CUDA_ERROR
}

