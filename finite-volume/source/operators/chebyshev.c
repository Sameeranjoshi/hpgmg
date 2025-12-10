//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// Based on Yousef Saad's Iterative Methods for Sparse Linear Algebra, Algorithm 12.1, page 399
//------------------------------------------------------------------------------------------------------------------------------
void smooth(level_type * level, int x_id, int rhs_id, REAL a, REAL b){
  if((CHEBYSHEV_DEGREE*NUM_SMOOTHS)&1){
    fprintf(stderr,"error... CHEBYSHEV_DEGREE*NUM_SMOOTHS must be even for the chebyshev smoother...\n");
    exit(0);
  }
  if( (level->dominant_eigenvalue_of_DinvA<=0.0) && (level->my_rank==0) )fprintf(stderr,"dominant_eigenvalue_of_DinvA <= 0.0 !\n");


  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  int s;
  int block;

  int compute_c1_c2 = 0;
  // allocate heap memory for coefficients
  if (level->chebyshev_c1 == NULL) { level->chebyshev_c1 = (REAL *)um_malloc(CHEBYSHEV_DEGREE * sizeof(REAL), level->um_access_policy); compute_c1_c2 = 1; }
  if (level->chebyshev_c2 == NULL) { level->chebyshev_c2 = (REAL *)um_malloc(CHEBYSHEV_DEGREE * sizeof(REAL), level->um_access_policy); compute_c1_c2 = 1; }

  // compute the Chebyshev coefficients...
  REAL beta     = 1.000*level->dominant_eigenvalue_of_DinvA;
//REAL alpha    = 0.300000*beta;
//REAL alpha    = 0.250000*beta;
//REAL alpha    = 0.166666*beta;
  REAL alpha    = 0.125000*beta;
  REAL theta    = 0.5*(beta+alpha);		// center of the spectral ellipse
  REAL delta    = 0.5*(beta-alpha);		// major axis?
  REAL sigma = theta/delta;
  REAL rho_n = 1/sigma;			// rho_0
#ifdef CUDA_UM_ALLOC
  REAL *chebyshev_c1 = level->chebyshev_c1;	// + c1*(x_n-x_nm1) == rho_n*rho_nm1
  REAL *chebyshev_c2 = level->chebyshev_c2;	// + c2*(b-Ax_n)
#else
  REAL chebyshev_c1[CHEBYSHEV_DEGREE];	// + c1*(x_n-x_nm1) == rho_n*rho_nm1
  REAL chebyshev_c2[CHEBYSHEV_DEGREE];	// + c2*(b-Ax_n)
#endif
  // compute coefficients only once if using gpu for this level
  if (!level->use_cuda || compute_c1_c2) {
    // make sure GPU is not running any tasks, as we cannot access managed memory concurrently on Kepler
    CUCHK( cudaDeviceSynchronize() );  			
    // now compute coefficients on cpu
    chebyshev_c1[0] = 0.0;
    chebyshev_c2[0] = 1/theta;
    for(s=1;s<CHEBYSHEV_DEGREE;s++){
      REAL rho_nm1 = rho_n;
      rho_n = 1.0/(2.0*sigma - rho_nm1);
      chebyshev_c1[s] = rho_n*rho_nm1;
      chebyshev_c2[s] = rho_n*2.0/delta;
    }
  }


  for(s=0;s<CHEBYSHEV_DEGREE*NUM_SMOOTHS;s++){
    // get ghost zone data... Chebyshev ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
   
    // apply the smoother... Chebyshev ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

    if (level->use_cuda) {
      cuda_smooth(*level, x_id, rhs_id, a, b, s, level->chebyshev_c1, level->chebyshev_c2);
    }
    else {
    PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
    for(block=0;block<level->num_my_blocks;block++){
      const int box = level->my_blocks[block].read.box;
      const int ilo = level->my_blocks[block].read.i;
      const int jlo = level->my_blocks[block].read.j;
      const int klo = level->my_blocks[block].read.k;
      const int ihi = level->my_blocks[block].dim.i + ilo;
      const int jhi = level->my_blocks[block].dim.j + jlo;
      const int khi = level->my_blocks[block].dim.k + klo;
      int i,j,k;
      const int ghosts = level->box_ghosts;
      const int jStride = level->my_boxes[box].jStride;
      const int kStride = level->my_boxes[box].kStride;
      const REAL h2inv = 1.0/(level->h*level->h);
      const REAL * __restrict__ rhs      = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
      const REAL * __restrict__ alpha    = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
      const REAL * __restrict__ beta_i   = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
      const REAL * __restrict__ beta_j   = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
      const REAL * __restrict__ beta_k   = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
      const REAL * __restrict__ Dinv     = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
      const REAL * __restrict__ valid    = level->my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); // cell is inside the domain

            REAL * __restrict__ x_np1;
      const REAL * __restrict__ x_n;
      const REAL * __restrict__ x_nm1;
                       if((s&1)==0){x_n    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                    x_nm1  = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride); 
                                    x_np1  = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                               else{x_n    = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                    x_nm1  = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); 
                                    x_np1  = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}
      const REAL c1 = chebyshev_c1[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.
      const REAL c2 = chebyshev_c2[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.

      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
        const int ijk = i + j*jStride + k*kStride;
        // According to Saad... but his was missing a Dinv[ijk] == D^{-1} !!!
        //  x_{n+1} = x_{n} + rho_{n} [ rho_{n-1}(x_{n} - x_{n-1}) + (2/delta)(b-Ax_{n}) ]
        //  x_temp[ijk] = x_n[ijk] + c1*(x_n[ijk]-x_temp[ijk]) + c2*Dinv[ijk]*(rhs[ijk]-Ax_n);
        const REAL Ax_n   = apply_op_ijk(x_n);
        const REAL lambda =     Dinv_ijk();
        x_np1[ijk] = x_n[ijk] + c1*(x_n[ijk]-x_nm1[ijk]) + c2*lambda*(rhs[ijk]-Ax_n);
      }}}

    } // box-loop
    } // use-cuda
    level->timers.smooth += (REAL)(getTime()-_timeStart);
  } // s-loop
}
