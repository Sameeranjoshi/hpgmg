//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include <omp.h>
float getTime(){
  return(omp_get_wtime()); // timers are in units of seconds; no conversion is necessary
}
