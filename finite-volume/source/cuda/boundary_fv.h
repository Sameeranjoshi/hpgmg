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

// Divide thread block into batches of threads (e.g. quads), each batch operates on one HPGMG tile/block
#define BLOCK_SIZE      128     // number of threads per thread block
#define NUM_BATCH       8       // mimber of batches per thread block
#undef  X
#define X(i)  __ldg(&x[i])

__constant__ int   faces[27] = {0,0,0,0,1,0,0,0,0,  0,1,0,1,0,1,0,1,0,  0,0,0,0,1,0,0,0,0};
__constant__ int   edges[27] = {0,1,0,1,0,1,0,1,0,  1,0,1,0,0,0,1,0,1,  0,1,0,1,0,1,0,1,0};
__constant__ int corners[27] = {1,0,1,0,0,0,1,0,1,  0,0,0,0,0,0,0,0,0,  1,0,1,0,0,0,1,0,1};

//------------------------------------------------------------------------------------------------------------------------------
template <int log_dim, int num_batch, int batch_size>
__global__ void apply_BCs_v1_kernel(level_type level, int x_id, int shape){
  // thread exit condition
  int batchid = blockIdx.x*num_batch + threadIdx.x/batch_size;
  if(batchid >= level.boundary_condition.num_blocks[shape]) return;

  // one CUDA thread block operates on 'batch_size' HPGMG tiles/blocks
  blockCopy_type block = level.boundary_condition.blocks[shape][batchid];

  float scale = 1.0f;
  if(  faces[block.subtype])scale=-1.0f;
  if(  edges[block.subtype])scale= 1.0f;
  if(corners[block.subtype])scale=-1.0f;

  int i,j,k;
  const int       box = block.read.box;
  const int     dim_i = block.dim.i;
  const int     dim_j = block.dim.j;
  const int     dim_k = block.dim.k;
  const int       ilo = block.read.i;
  const int       jlo = block.read.j;
  const int       klo = block.read.k;
  const int normal = 26-block.subtype; // invert the normal vector

  // hard code for box to box BC's
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  float * __restrict__  x = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride);

  // convert normal vector into pointer offsets...
  const int di = (((normal % 3)  )-1);
  const int dj = (((normal % 9)/3)-1);
  const int dk = (((normal / 9)  )-1);
  const int stride = di + dj*jStride + dk*kStride;

  for(int gid=threadIdx.x%batch_size; gid<dim_i*dim_j*dim_k; gid+=batch_size){
    k=(gid/dim_i)/dim_j;
    j=(gid/dim_i)%dim_j;
    i=gid%dim_i;
    int ijk = (i+ilo) + (j+jlo)*jStride + (k+klo)*kStride;
    x[ijk] = scale*x[ijk+stride]; // homogeneous linear = 1pt stencil
  }
}

//------------------------------------------------------------------------------------------------------------------------------
template <int num_batch, int batch_size>
__global__ void zero_ghost_region_kernel(level_type level, int x_id, int shape){
  // thread exit conditions
  int batchid = blockIdx.x*num_batch + threadIdx.x/batch_size;
  if(batchid >= level.boundary_condition.num_blocks[shape]) return;

  // one CUDA thread block operates on 'batch_size' HPGMG tiles/blocks
  blockCopy_type block = level.boundary_condition.blocks[shape][batchid];

  const int       box = block.read.box;
  const int     dim_i = block.dim.i;
  const int     dim_j = block.dim.j;
  const int     dim_k = block.dim.k;
  const int       ilo = block.read.i;
  const int       jlo = block.read.j;
  const int       klo = block.read.k;

  // hard code for box to box BC's
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  float * __restrict__  xn = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride); // physically the same, but use different pointers for read/write

  // zero out entire ghost region when not all points will be updated...
  for(int gid=threadIdx.x%batch_size; gid<dim_i*dim_j*dim_k; gid+=batch_size){
    int k=(gid/dim_i)/dim_j;
    int j=(gid/dim_i)%dim_j;
    int i=gid%dim_i;
    int ijk = (i+ilo) + (j+jlo)*jStride + (k+klo)*kStride;
    xn[ijk] = 0.0f;
  }
}

template <int log_dim, int num_batch, int batch_size>
__global__ void apply_BCs_v2_kernel(level_type level, int x_id, int shape){
  // thread exit conditon
  int batchid = blockIdx.x*num_batch + threadIdx.x/batch_size;
  if(batchid >= level.boundary_condition.num_blocks[shape]) return;

  // one CUDA thread block operates on 'batch_size' HPGMG tiles/blocks
  blockCopy_type block = level.boundary_condition.blocks[shape][batchid];

  const int box_dim    = level.box_dim;

  const int       box = block.read.box;
  const int     dim_i = block.dim.i;
  const int     dim_j = block.dim.j;
  const int     dim_k = block.dim.k;
  const int       ilo = block.read.i;
  const int       jlo = block.read.j;
  const int       klo = block.read.k;
  const int   subtype = block.subtype;
 
  // hard code for box to box BC's 
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  float * __restrict__  x  = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride);
  float * __restrict__  xn = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride); // physically the same, but use different pointers for read/write

  // apply the appropriate BC subtype (face, edge, corner)...
  if(faces[subtype]){
    //
    //    :.......:.......|.......:.......:.......:.
    //    :   -   :   ?   | -5/2  :  1/2  :   0   :
    //    :.......:.......|.......:.......:.......:.
    //
    int r=-1,rStride=-1,dim_r=-1,rlo=-1;
    int s=-1,sStride=-1,dim_s=-1,slo=-1;
    int t=-1,tStride=-1,dt=-1;

    // the two 4-point stencils can point in 6 different directions...
    switch(subtype){
      case  4:rlo=ilo;dim_r=dim_i;rStride=      1;slo=jlo;dim_s=dim_j;sStride=jStride;t=     -1;tStride=kStride;dt= tStride;break; // ij face, low k
      case 10:rlo=ilo;dim_r=dim_i;rStride=      1;slo=klo;dim_s=dim_k;sStride=kStride;t=     -1;tStride=jStride;dt= tStride;break; // ik face, low j
      case 12:rlo=jlo;dim_r=dim_j;rStride=jStride;slo=klo;dim_s=dim_k;sStride=kStride;t=     -1;tStride=      1;dt= tStride;break; // jk face, low i
      case 14:rlo=jlo;dim_r=dim_j;rStride=jStride;slo=klo;dim_s=dim_k;sStride=kStride;t=box_dim;tStride=      1;dt=-tStride;break; // jk face, high i
      case 16:rlo=ilo;dim_r=dim_i;rStride=      1;slo=klo;dim_s=dim_k;sStride=kStride;t=box_dim;tStride=jStride;dt=-tStride;break; // ik face, high j
      case 22:rlo=ilo;dim_r=dim_i;rStride=      1;slo=jlo;dim_s=dim_j;sStride=jStride;t=box_dim;tStride=kStride;dt=-tStride;break; // ij face, high k
    }

    // FIX... optimize for rStride==1 (unit-stride)
    for(int gid=threadIdx.x%batch_size; gid<dim_r*dim_s; gid+=batch_size){
      s=gid/dim_r;
      r=gid%dim_r;
      int ijk = (r+rlo)*rStride + (s+slo)*sStride + (t)*tStride;
      xn[ijk] = -2.5f*X(ijk+dt) + 0.5f*X(ijk+dt+dt);
    }
  }else
  if(edges[subtype]){
    //
    //          r   +---+---+ dt
    //          ^  /   /   /|/
    //          | +---+---+ |
    //          |/   /   /|/
    //      +---+---+---+ |
    //     /   /|   |   |/
    //    +---+ |---+---+---> ds
    //    |   |/   
    //    +---+    
    //
    int r=-1,rStride=-1,dim_r=-1,rlo=-1;
    int s=-1,sStride=-1,ds=-1;
    int t=-1,tStride=-1,dt=-1;
    // the four 16-point stencils (symmetry allows you to view it as 12 4-point) can point in 12 different directions...
    switch(subtype){
      case  1:rlo=ilo;dim_r=dim_i;rStride=      1;s=     -1;sStride=jStride;t=     -1;tStride=kStride;ds= sStride;dt= tStride;break; // i-edge,  low j,  low k
      case  3:rlo=jlo;dim_r=dim_j;rStride=jStride;s=     -1;sStride=      1;t=     -1;tStride=kStride;ds= sStride;dt= tStride;break; // j-edge,  low i,  low k
      case  5:rlo=jlo;dim_r=dim_j;rStride=jStride;s=box_dim;sStride=      1;t=     -1;tStride=kStride;ds=-sStride;dt= tStride;break; // j-edge, high i,  low k
      case  7:rlo=ilo;dim_r=dim_i;rStride=      1;s=box_dim;sStride=jStride;t=     -1;tStride=kStride;ds=-sStride;dt= tStride;break; // i-edge, high j,  low k
      case  9:rlo=klo;dim_r=dim_k;rStride=kStride;s=     -1;sStride=      1;t=     -1;tStride=jStride;ds= sStride;dt= tStride;break; // k-edge,  low i,  low j
      case 11:rlo=klo;dim_r=dim_k;rStride=kStride;s=box_dim;sStride=      1;t=     -1;tStride=jStride;ds=-sStride;dt= tStride;break; // k-edge, high i,  low j
      case 15:rlo=klo;dim_r=dim_k;rStride=kStride;s=     -1;sStride=      1;t=box_dim;tStride=jStride;ds= sStride;dt=-tStride;break; // k-edge,  low i, high j
      case 17:rlo=klo;dim_r=dim_k;rStride=kStride;s=box_dim;sStride=      1;t=box_dim;tStride=jStride;ds=-sStride;dt=-tStride;break; // k-edge, high i, high j
      case 19:rlo=ilo;dim_r=dim_i;rStride=      1;s=     -1;sStride=jStride;t=box_dim;tStride=kStride;ds= sStride;dt=-tStride;break; // i-edge,  low j, high k
      case 21:rlo=jlo;dim_r=dim_j;rStride=jStride;s=     -1;sStride=      1;t=box_dim;tStride=kStride;ds= sStride;dt=-tStride;break; // j-edge,  low i, high k
      case 23:rlo=jlo;dim_r=dim_j;rStride=jStride;s=box_dim;sStride=      1;t=box_dim;tStride=kStride;ds=-sStride;dt=-tStride;break; // j-edge, high i, high k
      case 25:rlo=ilo;dim_r=dim_i;rStride=      1;s=box_dim;sStride=jStride;t=box_dim;tStride=kStride;ds=-sStride;dt=-tStride;break; // i-edge, high j, high k
    }
    // FIX... optimize for rStride==1 (unit-stride)
    for(int gid=threadIdx.x%batch_size; gid<dim_r; gid+=batch_size){
      r=gid;
      int ijk = (r+rlo)*rStride + (s)*sStride + (t)*tStride;
      xn[ijk] =   6.25f*X(ijk+  ds+  dt) 
                - 1.25f*X(ijk+2*ds+  dt)
                - 1.25f*X(ijk+  ds+2*dt)
                + 0.25f*X(ijk+2*ds+2*dt);
    }
  }else
  if(corners[subtype]){
    //
    //                  +---+---+
    //                 /   /   /|
    //                +---+---+ |
    //               /   /   /|/|
    //              +---+---+ | | 
    //              |   |   |/|/
    //              +---+---+ |
    //              |   |   |/
    //          +---+---+---+
    //         /   /|
    //        +---+ |
    //        |   |/
    //        +---+
    //
    int i=-1,di=-1;
    int j=-1,dj=-1;
    int k=-1,dk=-1;
    // the eight 64-point stencils (symmetry allows you to view it as 56 4-point) can point in 8 different directions...
     switch(subtype){
      case  0:i=     -1;j=     -1;k=     -1;di= 1;dj= jStride;dk= kStride;break; //  low i,  low j,  low k
      case  2:i=box_dim;j=     -1;k=     -1;di=-1;dj= jStride;dk= kStride;break; // high i,  low j,  low k
      case  6:i=     -1;j=box_dim;k=     -1;di= 1;dj=-jStride;dk= kStride;break; //  low i, high j,  low k
      case  8:i=box_dim;j=box_dim;k=     -1;di=-1;dj=-jStride;dk= kStride;break; // high i, high j,  low k
      case 18:i=     -1;j=     -1;k=box_dim;di= 1;dj= jStride;dk=-kStride;break; //  low i,  low j, high k
      case 20:i=box_dim;j=     -1;k=box_dim;di=-1;dj= jStride;dk=-kStride;break; // high i,  low j, high k
      case 24:i=     -1;j=box_dim;k=box_dim;di= 1;dj=-jStride;dk=-kStride;break; //  low i, high j, high k
      case 26:i=box_dim;j=box_dim;k=box_dim;di=-1;dj=-jStride;dk=-kStride;break; // high i, high j, high k
    }
    if(threadIdx.x%batch_size>0) return;
    int ijk = (i) + (j)*jStride + (k)*kStride;
    xn[ijk] =  -15.625f*X(ijk+  di+  dj+  dk) 
               + 3.125f*X(ijk+2*di+  dj+  dk) 
               + 3.125f*X(ijk+  di+2*dj+  dk) 
               + 3.125f*X(ijk+  di+  dj+2*dk) 
               - 0.625f*X(ijk+2*di+2*dj+  dk) 
               - 0.625f*X(ijk+  di+2*dj+2*dk) 
               - 0.625f*X(ijk+2*di+  dj+2*dk) 
               + 0.125f*X(ijk+2*di+2*dj+2*dk);
  }
}

//------------------------------------------------------------------------------------------------------------------------------
template <int log_dim, int num_batch, int batch_size>
__global__ void apply_BCs_v4_kernel(level_type level, int x_id, int shape){
  // thread exit condition
  int bid = blockIdx.x*num_batch + threadIdx.x/batch_size;
  if(bid >= level.boundary_condition.num_blocks[shape]) return;

  // one CUDA thread block operates on 'batch_size' HPGMG tiles/blocks
  blockCopy_type block = level.boundary_condition.blocks[shape][bid];

  const int box_dim   = level.box_dim;
  const int       box = block.read.box; 
  const int     dim_i = block.dim.i;
  const int     dim_j = block.dim.j;
  const int     dim_k = block.dim.k;
  const int       ilo = block.read.i;
  const int       jlo = block.read.j;
  const int       klo = block.read.k;
  const int   subtype = block.subtype;
 
  // hard code for box to box BC's 
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  float * __restrict__  x  = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride);
  float * __restrict__  xn = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride); // physically the same, but use different pointers for read/write

  float OneTwelfth = 1.0f/12.0f;

    // apply the appropriate BC subtype (face, edge, corner)...
    if(faces[subtype]){
      //
      //    :....:....|....:....:....:....:.
      //    : ?? : ?? | x1 : x2 : x3 : x4 :
      //    :....:....|....:....:....:....:.
      //
      int r=-1,rStride=-1,dim_r=-1,rlo=-1;
      int s=-1,sStride=-1,dim_s=-1,slo=-1;
      int t=-1,tStride=-1,dt=-1;
    
      // the two 4-point stencils can point in 6 different directions...
      switch(subtype){
        case  4:rlo=ilo;dim_r=dim_i;rStride=      1;slo=jlo;dim_s=dim_j;sStride=jStride;t=     -1;tStride=kStride;dt= tStride;break; // ij face, low k
        case 10:rlo=ilo;dim_r=dim_i;rStride=      1;slo=klo;dim_s=dim_k;sStride=kStride;t=     -1;tStride=jStride;dt= tStride;break; // ik face, low j
        case 12:rlo=jlo;dim_r=dim_j;rStride=jStride;slo=klo;dim_s=dim_k;sStride=kStride;t=     -1;tStride=      1;dt= tStride;break; // jk face, low i
        case 14:rlo=jlo;dim_r=dim_j;rStride=jStride;slo=klo;dim_s=dim_k;sStride=kStride;t=box_dim;tStride=      1;dt=-tStride;break; // jk face, high i
        case 16:rlo=ilo;dim_r=dim_i;rStride=      1;slo=klo;dim_s=dim_k;sStride=kStride;t=box_dim;tStride=jStride;dt=-tStride;break; // ik face, high j
        case 22:rlo=ilo;dim_r=dim_i;rStride=      1;slo=jlo;dim_s=dim_j;sStride=jStride;t=box_dim;tStride=kStride;dt=-tStride;break; // ij face, high k
      }
      // FIX... optimize for rStride==1 (unit-stride)
      for(int gid=threadIdx.x%batch_size; gid<dim_r*dim_s; gid+=batch_size){
        s=gid/dim_r;
        r=gid%dim_r;
        int ijk = (r+rlo)*rStride + (s+slo)*sStride + (t)*tStride;
        float x1=X(ijk+dt), x2=X(ijk+2*dt), x3=X(ijk+3*dt), x4=X(ijk+4*dt);
        xn[ijk   ] = OneTwelfth*(  -77.0f*x1 +  43.0f*x2 -  17.0f*x3 +  3.0f*x4 );
        xn[ijk-dt] = OneTwelfth*( -505.0f*x1 + 335.0f*x2 - 145.0f*x3 + 27.0f*x4 );
      }
    }else
    if(edges[subtype]){
      //
      //                        +---+---+---+---+ dt
      //                       /   /   /   /   /|/
      //                      +---+---+---+---+ |
      //                r    /   /   /   /   /|/
      //                ^   +---+---+---+---+ |
      //                |  /   /   /   /   /|/
      //                | +---+---+---+---+ |
      //                |/   /   /   /   /|/
      //        +---+---+---+---+---+---+ |
      //       /   /   /|   |   |   |   |/
      //      +---+---+ |---+---+---+---+---> ds
      //     /   /   /|/ 
      //    +---+---+ |  
      //    |   |   |/   
      //    +---+---+    
      //
      //              ^ dt
      //              |
      //    :....:....|....:....:....:....:.
      //    : f4 : n4 | 14 : 24 : 34 : 44 :
      //    :....:....|....:....:....:....:.
      //    : f3 : n3 | 13 : 23 : 33 : 43 :
      //    :....:....|....:....:....:....:.
      //    : f2 : n2 | 12 : 22 : 32 : 42 :
      //    :....:....|....:....:....:....:.
      //    : f1 : n1 | 11 : 21 : 31 : 41 :
      //    ----------+---------------------> ds
      //    : ?? : ?? |
      //    :....:....|
      //    : ?? : ?? |
      //    :....:....|
      //
      int r=-1,rStride=-1,dim_r=-1,rlo=-1;
      int s=-1,sStride=-1,ds=-1;
      int t=-1,tStride=-1,dt=-1;
      // the four 16-point stencils (symmetry allows you to view it as 12 4-point) can point in 12 different directions...
      switch(subtype){
        case  1:rlo=ilo;dim_r=dim_i;rStride=      1;s=     -1;sStride=jStride;t=     -1;tStride=kStride;ds= sStride;dt= tStride;break; // i-edge,  low j,  low k
        case  3:rlo=jlo;dim_r=dim_j;rStride=jStride;s=     -1;sStride=      1;t=     -1;tStride=kStride;ds= sStride;dt= tStride;break; // j-edge,  low i,  low k
        case  5:rlo=jlo;dim_r=dim_j;rStride=jStride;s=box_dim;sStride=      1;t=     -1;tStride=kStride;ds=-sStride;dt= tStride;break; // j-edge, high i,  low k
        case  7:rlo=ilo;dim_r=dim_i;rStride=      1;s=box_dim;sStride=jStride;t=     -1;tStride=kStride;ds=-sStride;dt= tStride;break; // i-edge, high j,  low k
        case  9:rlo=klo;dim_r=dim_k;rStride=kStride;s=     -1;sStride=      1;t=     -1;tStride=jStride;ds= sStride;dt= tStride;break; // k-edge,  low i,  low j
        case 11:rlo=klo;dim_r=dim_k;rStride=kStride;s=box_dim;sStride=      1;t=     -1;tStride=jStride;ds=-sStride;dt= tStride;break; // k-edge, high i,  low j
        case 15:rlo=klo;dim_r=dim_k;rStride=kStride;s=     -1;sStride=      1;t=box_dim;tStride=jStride;ds= sStride;dt=-tStride;break; // k-edge,  low i, high j
        case 17:rlo=klo;dim_r=dim_k;rStride=kStride;s=box_dim;sStride=      1;t=box_dim;tStride=jStride;ds=-sStride;dt=-tStride;break; // k-edge, high i, high j
        case 19:rlo=ilo;dim_r=dim_i;rStride=      1;s=     -1;sStride=jStride;t=box_dim;tStride=kStride;ds= sStride;dt=-tStride;break; // i-edge,  low j, high k
        case 21:rlo=jlo;dim_r=dim_j;rStride=jStride;s=     -1;sStride=      1;t=box_dim;tStride=kStride;ds= sStride;dt=-tStride;break; // j-edge,  low i, high k
        case 23:rlo=jlo;dim_r=dim_j;rStride=jStride;s=box_dim;sStride=      1;t=box_dim;tStride=kStride;ds=-sStride;dt=-tStride;break; // j-edge, high i, high k
        case 25:rlo=ilo;dim_r=dim_i;rStride=      1;s=box_dim;sStride=jStride;t=box_dim;tStride=kStride;ds=-sStride;dt=-tStride;break; // i-edge, high j, high k
      }
      // FIX... optimize for rStride==1 (unit-stride)
      for(int gid=threadIdx.x%batch_size; gid<dim_r; gid+=batch_size){
	r=gid;
        int ijk = (r+rlo)*rStride + (s)*sStride + (t)*tStride;
        float x11 = X(ijk+  ds+  dt), x21 = X(ijk+2*ds+  dt), x31 = X(ijk+3*ds+  dt), x41 = X(ijk+4*ds+  dt);
        float x12 = X(ijk+  ds+2*dt), x22 = X(ijk+2*ds+2*dt), x32 = X(ijk+3*ds+2*dt), x42 = X(ijk+4*ds+2*dt);
        float x13 = X(ijk+  ds+3*dt), x23 = X(ijk+2*ds+3*dt), x33 = X(ijk+3*ds+3*dt), x43 = X(ijk+4*ds+3*dt);
        float x14 = X(ijk+  ds+4*dt), x24 = X(ijk+2*ds+4*dt), x34 = X(ijk+3*ds+4*dt), x44 = X(ijk+4*ds+4*dt);
            float n1 = OneTwelfth*(  -77.0f*x11 +  43.0f*x21 -  17.0f*x31 +  3.0f*x41 );
            float n2 = OneTwelfth*(  -77.0f*x12 +  43.0f*x22 -  17.0f*x32 +  3.0f*x42 );
            float n3 = OneTwelfth*(  -77.0f*x13 +  43.0f*x23 -  17.0f*x33 +  3.0f*x43 );
            float n4 = OneTwelfth*(  -77.0f*x14 +  43.0f*x24 -  17.0f*x34 +  3.0f*x44 );
            float f1 = OneTwelfth*( -505.0f*x11 + 335.0f*x21 - 145.0f*x31 + 27.0f*x41 );
            float f2 = OneTwelfth*( -505.0f*x12 + 335.0f*x22 - 145.0f*x32 + 27.0f*x42 );
            float f3 = OneTwelfth*( -505.0f*x13 + 335.0f*x23 - 145.0f*x33 + 27.0f*x43 );
            float f4 = OneTwelfth*( -505.0f*x14 + 335.0f*x24 - 145.0f*x34 + 27.0f*x44 );
        xn[ijk      ] = OneTwelfth*(  -77.0f*n1  +  43.0f*n2  -  17.0f*n3  +  3.0f*n4  );
        xn[ijk   -dt] = OneTwelfth*( -505.0f*n1  + 335.0f*n2  - 145.0f*n3  + 27.0f*n4  );
        xn[ijk-ds   ] = OneTwelfth*(  -77.0f*f1  +  43.0f*f2  -  17.0f*f3  +  3.0f*f4  );
        xn[ijk-ds-dt] = OneTwelfth*( -505.0f*f1  + 335.0f*f2  - 145.0f*f3  + 27.0f*f4  );
      }
    }else
    if(corners[subtype]){
      //
      //                        +---+---+---+---+
      //                       /   /   /   /   /|
      //                      +---+---+---+---+ |
      //                     /   /   /   /   /|/|
      //                    +---+---+---+---+ | |
      //                   /   /   /   /   /|/|/|
      //                  +---+---+---+---+ | | |
      //                 /   /   /   /   /|/|/|/|
      //                +---+---+---+---+ | | | |
      //                |   |   |   |   |/|/|/|/
      //                +---+---+---+---+ | | |
      //                |   |   |   |   |/|/|/
      //                +---+---+---+---+ | |
      //                |   |   |   |   |/|/
      //                +---+---+---+---+ |
      //                |   |   |   |   |/
      //        +---+---+---+---+---+---+
      //       /   /   /|
      //      +---+---+ |
      //     /   /   /|/|
      //    +---+---+ | |
      //    |   |   |/|/
      //    +---+---+ |
      //    |   |   |/
      //    +---+---+
      //
      int i=-1,di=-1;
      int j=-1,dj=-1;
      int k=-1,dk=-1;
      // the eight 64-point stencils (symmetry allows you to view it as 56 4-point) can point in 8 different directions...
      switch(subtype){
        case  0:i=     -1;j=     -1;k=     -1;di= 1;dj= jStride;dk= kStride;break; //  low i,  low j,  low k
        case  2:i=box_dim;j=     -1;k=     -1;di=-1;dj= jStride;dk= kStride;break; // high i,  low j,  low k
        case  6:i=     -1;j=box_dim;k=     -1;di= 1;dj=-jStride;dk= kStride;break; //  low i, high j,  low k
        case  8:i=box_dim;j=box_dim;k=     -1;di=-1;dj=-jStride;dk= kStride;break; // high i, high j,  low k
        case 18:i=     -1;j=     -1;k=box_dim;di= 1;dj= jStride;dk=-kStride;break; //  low i,  low j, high k
        case 20:i=box_dim;j=     -1;k=box_dim;di=-1;dj= jStride;dk=-kStride;break; // high i,  low j, high k
        case 24:i=     -1;j=box_dim;k=box_dim;di= 1;dj=-jStride;dk=-kStride;break; //  low i, high j, high k
        case 26:i=box_dim;j=box_dim;k=box_dim;di=-1;dj=-jStride;dk=-kStride;break; // high i, high j, high k
      }
      if(threadIdx.x%batch_size>0) return;
      int ijk = (i) + (j)*jStride + (k)*kStride;
      float x144 = X(ijk+  di+4*dj+4*dk);float x244 = X(ijk+2*di+4*dj+4*dk);float x344 = X(ijk+3*di+4*dj+4*dk);float x444 = X(ijk+4*di+4*dj+4*dk);
      float x134 = X(ijk+  di+3*dj+4*dk);float x234 = X(ijk+2*di+3*dj+4*dk);float x334 = X(ijk+3*di+3*dj+4*dk);float x434 = X(ijk+4*di+3*dj+4*dk);
      float x124 = X(ijk+  di+2*dj+4*dk);float x224 = X(ijk+2*di+2*dj+4*dk);float x324 = X(ijk+3*di+2*dj+4*dk);float x424 = X(ijk+4*di+2*dj+4*dk);
      float x114 = X(ijk+  di+  dj+4*dk);float x214 = X(ijk+2*di+  dj+4*dk);float x314 = X(ijk+3*di+  dj+4*dk);float x414 = X(ijk+4*di+  dj+4*dk);

      float x143 = X(ijk+  di+4*dj+3*dk);float x243 = X(ijk+2*di+4*dj+3*dk);float x343 = X(ijk+3*di+4*dj+3*dk);float x443 = X(ijk+4*di+4*dj+3*dk);
      float x133 = X(ijk+  di+3*dj+3*dk);float x233 = X(ijk+2*di+3*dj+3*dk);float x333 = X(ijk+3*di+3*dj+3*dk);float x433 = X(ijk+4*di+3*dj+3*dk);
      float x123 = X(ijk+  di+2*dj+3*dk);float x223 = X(ijk+2*di+2*dj+3*dk);float x323 = X(ijk+3*di+2*dj+3*dk);float x423 = X(ijk+4*di+2*dj+3*dk);
      float x113 = X(ijk+  di+  dj+3*dk);float x213 = X(ijk+2*di+  dj+3*dk);float x313 = X(ijk+3*di+  dj+3*dk);float x413 = X(ijk+4*di+  dj+3*dk);

      float x142 = X(ijk+  di+4*dj+2*dk);float x242 = X(ijk+2*di+4*dj+2*dk);float x342 = X(ijk+3*di+4*dj+2*dk);float x442 = X(ijk+4*di+4*dj+2*dk);
      float x132 = X(ijk+  di+3*dj+2*dk);float x232 = X(ijk+2*di+3*dj+2*dk);float x332 = X(ijk+3*di+3*dj+2*dk);float x432 = X(ijk+4*di+3*dj+2*dk);
      float x122 = X(ijk+  di+2*dj+2*dk);float x222 = X(ijk+2*di+2*dj+2*dk);float x322 = X(ijk+3*di+2*dj+2*dk);float x422 = X(ijk+4*di+2*dj+2*dk);
      float x112 = X(ijk+  di+  dj+2*dk);float x212 = X(ijk+2*di+  dj+2*dk);float x312 = X(ijk+3*di+  dj+2*dk);float x412 = X(ijk+4*di+  dj+2*dk);

      float x141 = X(ijk+  di+4*dj+  dk);float x241 = X(ijk+2*di+4*dj+  dk);float x341 = X(ijk+3*di+4*dj+  dk);float x441 = X(ijk+4*di+4*dj+  dk);
      float x131 = X(ijk+  di+3*dj+  dk);float x231 = X(ijk+2*di+3*dj+  dk);float x331 = X(ijk+3*di+3*dj+  dk);float x431 = X(ijk+4*di+3*dj+  dk);
      float x121 = X(ijk+  di+2*dj+  dk);float x221 = X(ijk+2*di+2*dj+  dk);float x321 = X(ijk+3*di+2*dj+  dk);float x421 = X(ijk+4*di+2*dj+  dk);
      float x111 = X(ijk+  di+  dj+  dk);float x211 = X(ijk+2*di+  dj+  dk);float x311 = X(ijk+3*di+  dj+  dk);float x411 = X(ijk+4*di+  dj+  dk);

      // 32 stencils in i...
      float n11 = OneTwelfth*(  -77.0f*x111 +  43.0f*x211 -  17.0f*x311 +  3.0f*x411 );
      float n21 = OneTwelfth*(  -77.0f*x121 +  43.0f*x221 -  17.0f*x321 +  3.0f*x421 );
      float n31 = OneTwelfth*(  -77.0f*x131 +  43.0f*x231 -  17.0f*x331 +  3.0f*x431 );
      float n41 = OneTwelfth*(  -77.0f*x141 +  43.0f*x241 -  17.0f*x341 +  3.0f*x441 );
      float n12 = OneTwelfth*(  -77.0f*x112 +  43.0f*x212 -  17.0f*x312 +  3.0f*x412 );
      float n22 = OneTwelfth*(  -77.0f*x122 +  43.0f*x222 -  17.0f*x322 +  3.0f*x422 );
      float n32 = OneTwelfth*(  -77.0f*x132 +  43.0f*x232 -  17.0f*x332 +  3.0f*x432 );
      float n42 = OneTwelfth*(  -77.0f*x142 +  43.0f*x242 -  17.0f*x342 +  3.0f*x442 );
      float n13 = OneTwelfth*(  -77.0f*x113 +  43.0f*x213 -  17.0f*x313 +  3.0f*x413 );
      float n23 = OneTwelfth*(  -77.0f*x123 +  43.0f*x223 -  17.0f*x323 +  3.0f*x423 );
      float n33 = OneTwelfth*(  -77.0f*x133 +  43.0f*x233 -  17.0f*x333 +  3.0f*x433 );
      float n43 = OneTwelfth*(  -77.0f*x143 +  43.0f*x243 -  17.0f*x343 +  3.0f*x443 );
      float n14 = OneTwelfth*(  -77.0f*x114 +  43.0f*x214 -  17.0f*x314 +  3.0f*x414 );
      float n24 = OneTwelfth*(  -77.0f*x124 +  43.0f*x224 -  17.0f*x324 +  3.0f*x424 );
      float n34 = OneTwelfth*(  -77.0f*x134 +  43.0f*x234 -  17.0f*x334 +  3.0f*x434 );
      float n44 = OneTwelfth*(  -77.0f*x144 +  43.0f*x244 -  17.0f*x344 +  3.0f*x444 );

      float f11 = OneTwelfth*( -505.0f*x111 + 335.0f*x211 - 145.0f*x311 +  27.0f*x411 );
      float f21 = OneTwelfth*( -505.0f*x121 + 335.0f*x221 - 145.0f*x321 +  27.0f*x421 );
      float f31 = OneTwelfth*( -505.0f*x131 + 335.0f*x231 - 145.0f*x331 +  27.0f*x431 );
      float f41 = OneTwelfth*( -505.0f*x141 + 335.0f*x241 - 145.0f*x341 +  27.0f*x441 );
      float f12 = OneTwelfth*( -505.0f*x112 + 335.0f*x212 - 145.0f*x312 +  27.0f*x412 );
      float f22 = OneTwelfth*( -505.0f*x122 + 335.0f*x222 - 145.0f*x322 +  27.0f*x422 );
      float f32 = OneTwelfth*( -505.0f*x132 + 335.0f*x232 - 145.0f*x332 +  27.0f*x432 );
      float f42 = OneTwelfth*( -505.0f*x142 + 335.0f*x242 - 145.0f*x342 +  27.0f*x442 );
      float f13 = OneTwelfth*( -505.0f*x113 + 335.0f*x213 - 145.0f*x313 +  27.0f*x413 );
      float f23 = OneTwelfth*( -505.0f*x123 + 335.0f*x223 - 145.0f*x323 +  27.0f*x423 );
      float f33 = OneTwelfth*( -505.0f*x133 + 335.0f*x233 - 145.0f*x333 +  27.0f*x433 );
      float f43 = OneTwelfth*( -505.0f*x143 + 335.0f*x243 - 145.0f*x343 +  27.0f*x443 );
      float f14 = OneTwelfth*( -505.0f*x114 + 335.0f*x214 - 145.0f*x314 +  27.0f*x414 );
      float f24 = OneTwelfth*( -505.0f*x124 + 335.0f*x224 - 145.0f*x324 +  27.0f*x424 );
      float f34 = OneTwelfth*( -505.0f*x134 + 335.0f*x234 - 145.0f*x334 +  27.0f*x434 );
      float f44 = OneTwelfth*( -505.0f*x144 + 335.0f*x244 - 145.0f*x344 +  27.0f*x444 );

      // 16 stencils in j...
      float nn1 = OneTwelfth*(  -77.0f*n11 +  43.0f*n21 -  17.0f*n31 +  3.0f*n41 );
      float nn2 = OneTwelfth*(  -77.0f*n12 +  43.0f*n22 -  17.0f*n32 +  3.0f*n42 );
      float nn3 = OneTwelfth*(  -77.0f*n13 +  43.0f*n23 -  17.0f*n33 +  3.0f*n43 );
      float nn4 = OneTwelfth*(  -77.0f*n14 +  43.0f*n24 -  17.0f*n34 +  3.0f*n44 );
      float nf1 = OneTwelfth*( -505.0f*n11 + 335.0f*n21 - 145.0f*n31 + 27.0f*n41 );
      float nf2 = OneTwelfth*( -505.0f*n12 + 335.0f*n22 - 145.0f*n32 + 27.0f*n42 );
      float nf3 = OneTwelfth*( -505.0f*n13 + 335.0f*n23 - 145.0f*n33 + 27.0f*n43 );
      float nf4 = OneTwelfth*( -505.0f*n14 + 335.0f*n24 - 145.0f*n34 + 27.0f*n44 );

      float fn1 = OneTwelfth*(  -77.0f*f11 +  43.0f*f21 -  17.0f*f31 +  3.0f*f41 );
      float fn2 = OneTwelfth*(  -77.0f*f12 +  43.0f*f22 -  17.0f*f32 +  3.0f*f42 );
      float fn3 = OneTwelfth*(  -77.0f*f13 +  43.0f*f23 -  17.0f*f33 +  3.0f*f43 );
      float fn4 = OneTwelfth*(  -77.0f*f14 +  43.0f*f24 -  17.0f*f34 +  3.0f*f44 );
      float ff1 = OneTwelfth*( -505.0f*f11 + 335.0f*f21 - 145.0f*f31 + 27.0f*f41 );
      float ff2 = OneTwelfth*( -505.0f*f12 + 335.0f*f22 - 145.0f*f32 + 27.0f*f42 );
      float ff3 = OneTwelfth*( -505.0f*f13 + 335.0f*f23 - 145.0f*f33 + 27.0f*f43 );
      float ff4 = OneTwelfth*( -505.0f*f14 + 335.0f*f24 - 145.0f*f34 + 27.0f*f44 );

      //  8 stencils in k...
      float nnn = OneTwelfth*(  -77.0f*nn1 +  43.0f*nn2 -  17.0f*nn3 +  3.0f*nn4 );
      float nnf = OneTwelfth*( -505.0f*nn1 + 335.0f*nn2 - 145.0f*nn3 + 27.0f*nn4 );
      float nfn = OneTwelfth*(  -77.0f*nf1 +  43.0f*nf2 -  17.0f*nf3 +  3.0f*nf4 );
      float nff = OneTwelfth*( -505.0f*nf1 + 335.0f*nf2 - 145.0f*nf3 + 27.0f*nf4 );
      float fnn = OneTwelfth*(  -77.0f*fn1 +  43.0f*fn2 -  17.0f*fn3 +  3.0f*fn4 );
      float fnf = OneTwelfth*( -505.0f*fn1 + 335.0f*fn2 - 145.0f*fn3 + 27.0f*fn4 );
      float ffn = OneTwelfth*(  -77.0f*ff1 +  43.0f*ff2 -  17.0f*ff3 +  3.0f*ff4 );
      float fff = OneTwelfth*( -505.0f*ff1 + 335.0f*ff2 - 145.0f*ff3 + 27.0f*ff4 );

      // commit to the 8 ghost zones in this corner...
      xn[ijk         ] = nnn;
      xn[ijk      -dk] = nnf;
      xn[ijk   -dj   ] = nfn;
      xn[ijk   -dj-dk] = nff;
      xn[ijk-di      ] = fnn;
      xn[ijk-di   -dk] = fnf;
      xn[ijk-di-dj   ] = ffn;
      xn[ijk-di-dj-dk] = fff;
    }
}

//------------------------------------------------------------------------------------------------------------------------------
template <int log_dim, int num_batch, int batch_size>
__global__ void extrapolate_betas_kernel(level_type level, int shape){
  // thread exit condition
  int bid = blockIdx.x*num_batch + threadIdx.x/batch_size;
  if(bid >= level.boundary_condition.num_blocks[shape]) return;

  // one CUDA thread block operates on 'batch_size' HPGMG tiles/blocks
  blockCopy_type block = level.boundary_condition.blocks[shape][bid];

  int i,j,k;
  const int       box = block.read.box; 
  const int     dim_i = block.dim.i;
  const int     dim_j = block.dim.j;
  const int     dim_k = block.dim.k;
  const int       ilo = block.read.i;
  const int       jlo = block.read.j;
  const int       klo = block.read.k;

  // total hack/reuse of the existing boundary list...
  //   however, whereas boundary subtype represents the normal to the domain at that point, 
  //   one needs the box-relative (not domain-relative) normal when extending the face averaged beta's into the ghost zones
  //   Thus, I reuse the list to tell me which areas are beyond the domain boundary, but must calculate their normals here
  int   subtype = 13;
  if(ilo <               0)subtype-=1;
  if(jlo <               0)subtype-=3;
  if(klo <               0)subtype-=9;
  if(ilo >= level.box_dim)subtype+=1;
  if(jlo >= level.box_dim)subtype+=3;
  if(klo >= level.box_dim)subtype+=9;
  const int    normal = 26-subtype; // invert the normal vector
 
  // hard code for box to box BC's 
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  float * __restrict__  beta_i = level.my_boxes[box].vectors[VECTOR_BETA_I] + level.my_boxes[box].ghosts*(1+jStride+kStride);
  float * __restrict__  beta_j = level.my_boxes[box].vectors[VECTOR_BETA_J] + level.my_boxes[box].ghosts*(1+jStride+kStride);
  float * __restrict__  beta_k = level.my_boxes[box].vectors[VECTOR_BETA_K] + level.my_boxes[box].ghosts*(1+jStride+kStride);

  // convert normal vector into pointer offsets...
  const int di = (((normal % 3)  )-1);
  const int dj = (((normal % 9)/3)-1);
  const int dk = (((normal / 9)  )-1);

  // beta_i should be extrapolated in the j- and k-directions, but not i
  // beta_j should be extrapolated in the i- and k-directions, but not j
  // beta_k should be extrapolated in the i- and j-directions, but not k
  // e.g.
  //                  .................................
  //                 .       .       .       .       .
  //                .       .  ???  .  ???  .       .
  //               .       .       .       .       .
  //              ........+-------+-------+........
  //             .       /       /       /       .
  //            .  ???  /<betaK>/<betaK>/  ???  .
  //           .       /       /       /       .
  //          ........+-------+-------+........
  //         .       /       /       /       .
  //        .  ???  /<betaK>/<betaK>/  ???  .
  //       .       /       /       /       .
  //      ........+-------+-------+........   k   j
  //     .       .       .       .       .    ^  ^   
  //    .       .  ???  .  ???  .       .     | /
  //   .       .       .       .       .      |/
  //  .................................       +-----> i
  //
  const int biStride =      dj*jStride + dk*kStride;
  const int bjStride = di              + dk*kStride;
  const int bkStride = di + dj*jStride             ;

  // note, 
  //   the face values normal to i should have been filled via RESTRICT_I (skip them)
  //   the face values normal to j should have been filled via RESTRICT_J (skip them)
  //   the face values normal to k should have been filled via RESTRICT_K (skip them)
  if(level.box_dim>=5){
    // quartic extrapolation... 
    for(int gid=threadIdx.x%batch_size; gid<dim_i*dim_j*dim_k; gid+=batch_size){
      k=(gid/dim_i)/dim_j;
      j=(gid/dim_i)%dim_j;
      i=gid%dim_i;
      int ijk = (i+ilo) + (j+jlo)*jStride + (k+klo)*kStride;
      if( (subtype!=14) && (subtype!=12) ){beta_i[ijk] = 5.0f*beta_i[ijk+biStride] - 10.0f*beta_i[ijk+2*biStride] + 10.0f*beta_i[ijk+3*biStride] - 5.0f*beta_i[ijk+4*biStride] + beta_i[ijk+5*biStride];}
      if( (subtype!=16) && (subtype!=10) ){beta_j[ijk] = 5.0f*beta_j[ijk+bjStride] - 10.0f*beta_j[ijk+2*bjStride] + 10.0f*beta_j[ijk+3*bjStride] - 5.0f*beta_j[ijk+4*bjStride] + beta_j[ijk+5*bjStride];}
      if( (subtype!=22) && (subtype!= 4) ){beta_k[ijk] = 5.0f*beta_k[ijk+bkStride] - 10.0f*beta_k[ijk+2*bkStride] + 10.0f*beta_k[ijk+3*bkStride] - 5.0f*beta_k[ijk+4*bkStride] + beta_k[ijk+5*bkStride];}
    }
  }else 
  if(level.box_dim>=4){
    // cubic extrapolation... 
    for(int gid=threadIdx.x%batch_size; gid<dim_i*dim_j*dim_k; gid+=batch_size){
      k=(gid/dim_i)/dim_j;
      j=(gid/dim_i)%dim_j;
      i=gid%dim_i;
      int ijk = (i+ilo) + (j+jlo)*jStride + (k+klo)*kStride;
      if( (subtype!=14) && (subtype!=12) ){beta_i[ijk] = 4.0f*beta_i[ijk+biStride] - 6.0f*beta_i[ijk+2*biStride] + 4.0f*beta_i[ijk+3*biStride] - beta_i[ijk+4*biStride];}
      if( (subtype!=16) && (subtype!=10) ){beta_j[ijk] = 4.0f*beta_j[ijk+bjStride] - 6.0f*beta_j[ijk+2*bjStride] + 4.0f*beta_j[ijk+3*bjStride] - beta_j[ijk+4*bjStride];}
      if( (subtype!=22) && (subtype!= 4) ){beta_k[ijk] = 4.0f*beta_k[ijk+bkStride] - 6.0f*beta_k[ijk+2*bkStride] + 4.0f*beta_k[ijk+3*bkStride] - beta_k[ijk+4*bkStride];}
    }
  }else 
  if(level.box_dim>=2){
    // linear extrapolation...
    for(int gid=threadIdx.x%batch_size; gid<dim_i*dim_j*dim_k; gid+=batch_size){
      k=(gid/dim_i)/dim_j;
      j=(gid/dim_i)%dim_j;
      i=gid%dim_i;
      int ijk = (i+ilo) + (j+jlo)*jStride + (k+klo)*kStride;
      if( (subtype!=14) && (subtype!=12) ){beta_i[ijk] = 2.0f*beta_i[ijk+biStride] - beta_i[ijk+2*biStride];}
      if( (subtype!=16) && (subtype!=10) ){beta_j[ijk] = 2.0f*beta_j[ijk+bjStride] - beta_j[ijk+2*bjStride];}
      if( (subtype!=22) && (subtype!= 4) ){beta_k[ijk] = 2.0f*beta_k[ijk+bkStride] - beta_k[ijk+2*bkStride];}
    }
  }
}
#undef  KERNEL
#define KERNEL(log_dim, shape) \
  apply_BCs_v1_kernel<log_dim,NUM_BATCH,(BLOCK_SIZE/NUM_BATCH)><<<grid,block>>>(level,x_id,shape);

extern "C"
void cuda_apply_BCs_v1(level_type level, int x_id, int shape)
{
  int block = BLOCK_SIZE;
  int grid = (level.boundary_condition.num_blocks[shape]+NUM_BATCH-1)/NUM_BATCH; 
  if(grid<=0) return;

  int log_dim = (int)log2f((float)level.dim.i);
  KERNEL_LEVEL(log_dim, shape);
  CUDA_ERROR
}
#undef  KERNEL
#define KERNEL(log_dim, shape) \
  apply_BCs_v2_kernel<log_dim,NUM_BATCH,(BLOCK_SIZE/NUM_BATCH)><<<grid,block>>>(level,x_id,shape);

extern "C"
void cuda_apply_BCs_v2(level_type level, int x_id, int shape)
{
  int block = BLOCK_SIZE;
  int grid = (level.boundary_condition.num_blocks[shape]+NUM_BATCH-1)/NUM_BATCH; 
  if(grid<=0) return;

  if(level.box_ghosts>1){
    zero_ghost_region_kernel<NUM_BATCH,(BLOCK_SIZE/NUM_BATCH)><<<grid,block>>>(level,x_id,shape);
    CUDA_ERROR
  }

  int log_dim = (int)log2f((float)level.dim.i);
  KERNEL_LEVEL(log_dim, shape);
  CUDA_ERROR
}
#undef  KERNEL
#define KERNEL(log_dim, shape) \
  apply_BCs_v4_kernel<log_dim,NUM_BATCH,(BLOCK_SIZE/NUM_BATCH)><<<grid,block>>>(level,x_id,shape);

extern "C"
void cuda_apply_BCs_v4(level_type level, int x_id, int shape)
{
  int block = BLOCK_SIZE;
  int grid = (level.boundary_condition.num_blocks[shape]+NUM_BATCH-1)/NUM_BATCH; 
  if(grid<=0) return;

  if(level.box_ghosts>1){
    zero_ghost_region_kernel<NUM_BATCH,(BLOCK_SIZE/NUM_BATCH)><<<grid,block>>>(level,x_id,shape);
    CUDA_ERROR
  }

  int log_dim = (int)log2f((float)level.dim.i);
  KERNEL_LEVEL(log_dim, shape);
  CUDA_ERROR
}
#undef  KERNEL
#define KERNEL(log_dim, shape) \
  extrapolate_betas_kernel<log_dim,NUM_BATCH,(BLOCK_SIZE/NUM_BATCH)><<<grid,block>>>(level,shape);

extern "C"
void cuda_extrapolate_betas(level_type level, int shape)
{
  int block = BLOCK_SIZE;
  int grid = (level.boundary_condition.num_blocks[shape]+NUM_BATCH-1)/NUM_BATCH; 
  if(grid<=0) return;

  int log_dim = (int)log2f((float)level.dim.i);
  KERNEL_LEVEL(log_dim, shape);
  CUDA_ERROR
}
