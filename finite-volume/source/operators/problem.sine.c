//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846f // in case math.h doesn't define it
#endif
void evaluateBeta(float x, float y, float z, float *B, float *Bx, float *By, float *Bz){
  float Bmin = 1.0f;
  float Bmax = 10.0f;
  float c2 = (Bmax-Bmin)/2; // coefficients to affect this transition
  float c1 = (Bmax+Bmin)/2;
  float c3 = 10.0f;          // how sharply (B)eta transitions
  float xcenter = 0.50f;
  float ycenter = 0.50f;
  float zcenter = 0.50f;
  // calculate distance from center of the domain (0.5f,0.5f,0.5f)
  float r2   = powf((x-xcenter),2) +  powf((y-ycenter),2) +  powf((z-zcenter),2);
  float r2x  = 2.0f*(x-xcenter);
  float r2y  = 2.0f*(y-ycenter);
  float r2z  = 2.0f*(z-zcenter);
//float r2xx = 2.0f;
//float r2yy = 2.0f;
//float r2zz = 2.0f;
  float r    = powf(r2,0.5f);
  float rx   = 0.5f*r2x*powf(r2,-0.5f);
  float ry   = 0.5f*r2y*powf(r2,-0.5f);
  float rz   = 0.5f*r2z*powf(r2,-0.5f);
//float rxx  = 0.5f*r2xx*powf(r2,-0.5f) - 0.25f*r2x*r2x*powf(r2,-1.5f);
//float ryy  = 0.5f*r2yy*powf(r2,-0.5f) - 0.25f*r2y*r2y*powf(r2,-1.5f);
//float rzz  = 0.5f*r2zz*powf(r2,-0.5f) - 0.25f*r2z*r2z*powf(r2,-1.5f);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  *B  =           c1+c2*tanh( c3*(r-0.25f) );
  *Bx = c2*c3*rx*(1-powf(tanh( c3*(r-0.25f) ),2));
  *By = c2*c3*ry*(1-powf(tanh( c3*(r-0.25f) ),2));
  *Bz = c2*c3*rz*(1-powf(tanh( c3*(r-0.25f) ),2));
}


//------------------------------------------------------------------------------------------------------------------------------
void evaluateU(float x, float y, float z, float *U, float *Ux, float *Uy, float *Uz, float *Uxx, float *Uyy, float *Uzz, int isPeriodic){
  float c1 = 2.0f*M_PI;
  float c2 = 6.0f*M_PI;
  float p = 13; // must be odd(?) and allows up to p-2 order MG
        *U    =                                                       powf(sinf(c1*x),p  )*powf(sinf(c1*y),p)*powf(sinf(c1*z),p);
        *Ux   =                                        c1*p*cosf(c1*x)*powf(sinf(c1*x),p-1)*powf(sinf(c1*y),p)*powf(sinf(c1*z),p);
        *Uy   =                                        c1*p*cosf(c1*y)*powf(sinf(c1*y),p-1)*powf(sinf(c1*x),p)*powf(sinf(c1*z),p);
        *Uz   =                                        c1*p*cosf(c1*z)*powf(sinf(c1*z),p-1)*powf(sinf(c1*x),p)*powf(sinf(c1*y),p);
        *Uxx  = c1*c1*p*( (p-1)*powf(sinf(c1*x),p-2)*powf(cosf(c1*x),2) - powf(sinf(c1*x),p) )*powf(sinf(c1*y),p)*powf(sinf(c1*z),p);
        *Uyy  = c1*c1*p*( (p-1)*powf(sinf(c1*y),p-2)*powf(cosf(c1*y),2) - powf(sinf(c1*y),p) )*powf(sinf(c1*x),p)*powf(sinf(c1*z),p);
        *Uzz  = c1*c1*p*( (p-1)*powf(sinf(c1*z),p-2)*powf(cosf(c1*z),2) - powf(sinf(c1*z),p) )*powf(sinf(c1*x),p)*powf(sinf(c1*y),p);

        *U   +=                                                       powf(sinf(c2*x),p  )*powf(sinf(c2*y),p)*powf(sinf(c2*z),p);
        *Ux  +=                                        c2*p*cosf(c2*x)*powf(sinf(c2*x),p-1)*powf(sinf(c2*y),p)*powf(sinf(c2*z),p);
        *Uy  +=                                        c2*p*cosf(c2*y)*powf(sinf(c2*y),p-1)*powf(sinf(c2*x),p)*powf(sinf(c2*z),p);
        *Uz  +=                                        c2*p*cosf(c2*z)*powf(sinf(c2*z),p-1)*powf(sinf(c2*x),p)*powf(sinf(c2*y),p);
        *Uxx += c2*c2*p*( (p-1)*powf(sinf(c2*x),p-2)*powf(cosf(c2*x),2) - powf(sinf(c2*x),p) )*powf(sinf(c2*y),p)*powf(sinf(c2*z),p);
        *Uyy += c2*c2*p*( (p-1)*powf(sinf(c2*y),p-2)*powf(cosf(c2*y),2) - powf(sinf(c2*y),p) )*powf(sinf(c2*x),p)*powf(sinf(c2*z),p);
        *Uzz += c2*c2*p*( (p-1)*powf(sinf(c2*z),p-2)*powf(cosf(c2*z),2) - powf(sinf(c2*z),p) )*powf(sinf(c2*x),p)*powf(sinf(c2*y),p);
}


//------------------------------------------------------------------------------------------------------------------------------
void initialize_problem(level_type * level, float hLevel, float a, float b){
  level->h = hLevel;

  int box;
  for(box=0;box<level->num_my_boxes;box++){
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    const int   dim_i = level->my_boxes[box].dim;
    const int   dim_j = level->my_boxes[box].dim;
    const int   dim_k = level->my_boxes[box].dim;
    #ifdef _OPENMP
    #pragma omp parallel for private(k,j,i) collapse(3)
    #endif
    for(k=0;k<=dim_k;k++){ // include high face
    for(j=0;j<=dim_j;j++){ // include high face
    for(i=0;i<=dim_i;i++){ // include high face
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      int ijk = (i+ghosts) + (j+ghosts)*jStride + (k+ghosts)*kStride;
      float x = hLevel*( (float)(i+level->my_boxes[box].low.i) + 0.5f ); // +0.5f to get to the center of cell
      float y = hLevel*( (float)(j+level->my_boxes[box].low.j) + 0.5f );
      float z = hLevel*( (float)(k+level->my_boxes[box].low.k) + 0.5f );
      float A,B,Bx,By,Bz,Bi,Bj,Bk;
      float U,Ux,Uy,Uz,Uxx,Uyy,Uzz;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      A  = 1.0f;
      B  = 1.0f;
      Bx = 0.0f;
      By = 0.0f;
      Bz = 0.0f; 
      Bi = 1.0f;
      Bj = 1.0f;
      Bk = 1.0f;
      #ifdef STENCIL_VARIABLE_COEFFICIENT // variable coefficient problem...
      evaluateBeta(x-hLevel*0.5f,y           ,z           ,&Bi,&Bx,&By,&Bz); // face-centered value of Beta for beta_i
      evaluateBeta(x           ,y-hLevel*0.5f,z           ,&Bj,&Bx,&By,&Bz); // face-centered value of Beta for beta_j
      evaluateBeta(x           ,y           ,z-hLevel*0.5f,&Bk,&Bx,&By,&Bz); // face-centered value of Beta for beta_k
      evaluateBeta(x           ,y           ,z           ,&B ,&Bx,&By,&Bz); // cell-centered value of Beta
      #endif
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      evaluateU(x,y,z,&U,&Ux,&Uy,&Uz,&Uxx,&Uyy,&Uzz, (level->boundary_condition.type == BC_PERIODIC) );
      float F = a*A*U - b*( (Bx*Ux + By*Uy + Bz*Uz)  +  B*(Uxx + Uyy + Uzz) );
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      level->my_boxes[box].vectors[VECTOR_BETA_I][ijk] = Bi;
      level->my_boxes[box].vectors[VECTOR_BETA_J][ijk] = Bj;
      level->my_boxes[box].vectors[VECTOR_BETA_K][ijk] = Bk;
      level->my_boxes[box].vectors[VECTOR_ALPHA ][ijk] = A;
      level->my_boxes[box].vectors[VECTOR_UTRUE ][ijk] = U;
      level->my_boxes[box].vectors[VECTOR_F     ][ijk] = F;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    }}}
  }

}
//------------------------------------------------------------------------------------------------------------------------------
