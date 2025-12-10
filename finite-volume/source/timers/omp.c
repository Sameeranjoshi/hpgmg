//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include <omp.h>
// Timer should return double for precision, cast to REAL only when storing
double getTime(){
  return(omp_get_wtime()); // timers are in units of seconds; no conversion is necessary
}
