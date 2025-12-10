//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef TIMER_H
#define TIMER_H

  #include<stdint.h>
  #include "defines.h"  // For REAL type definition

  #ifdef _OPENMP
    #include <omp.h>
    // Keep timer in double precision to avoid precision loss with float32
    #define getTime() ((double)omp_get_wtime())

  #elif USE_MPI
    #include <mpi.h>
    // Keep timer in double precision to avoid precision loss with float32
    #define getTime() ((double)MPI_Wtime())

  #else
    // user must provide a function getTime and include it in timers.c
    // if calibration is necesary, then the user must #define CALIBRATE_TIMER
    // Timer should return double for precision
    double getTime();
  #endif

#endif
