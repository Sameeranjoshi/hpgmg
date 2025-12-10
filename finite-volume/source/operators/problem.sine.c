//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846 // in case math.h doesn't define it
#endif
void evaluateBeta(REAL x, REAL y, REAL z, REAL *B, REAL *Bx, REAL *By, REAL *Bz){
  REAL Bmin =  1.0;
  REAL Bmax = 10.0;
  REAL c2 = (Bmax-Bmin)/2; // coefficients to affect this transition
  REAL c1 = (Bmax+Bmin)/2;
  REAL c3 = 10.0;          // how sharply (B)eta transitions
  REAL xcenter = 0.50;
  REAL ycenter = 0.50;
  REAL zcenter = 0.50;
  // calculate distance from center of the domain (0.5,0.5,0.5)
  REAL r2   = pow((x-xcenter),2) +  pow((y-ycenter),2) +  pow((z-zcenter),2);
  REAL r2x  = 2.0*(x-xcenter);
  REAL r2y  = 2.0*(y-ycenter);
  REAL r2z  = 2.0*(z-zcenter);
//REAL r2xx = 2.0;
//REAL r2yy = 2.0;
//REAL r2zz = 2.0;
  REAL r    = pow(r2,0.5);
  REAL rx   = 0.5*r2x*pow(r2,-0.5);
  REAL ry   = 0.5*r2y*pow(r2,-0.5);
  REAL rz   = 0.5*r2z*pow(r2,-0.5);
//REAL rxx  = 0.5*r2xx*pow(r2,-0.5) - 0.25*r2x*r2x*pow(r2,-1.5);
//REAL ryy  = 0.5*r2yy*pow(r2,-0.5) - 0.25*r2y*r2y*pow(r2,-1.5);
//REAL rzz  = 0.5*r2zz*pow(r2,-0.5) - 0.25*r2z*r2z*pow(r2,-1.5);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  *B  =           c1+c2*tanh( c3*(r-0.25) );
  *Bx = c2*c3*rx*(1-pow(tanh( c3*(r-0.25) ),2));
  *By = c2*c3*ry*(1-pow(tanh( c3*(r-0.25) ),2));
  *Bz = c2*c3*rz*(1-pow(tanh( c3*(r-0.25) ),2));
}


//------------------------------------------------------------------------------------------------------------------------------
void evaluateU(REAL x, REAL y, REAL z, REAL *U, REAL *Ux, REAL *Uy, REAL *Uz, REAL *Uxx, REAL *Uyy, REAL *Uzz, int isPeriodic){
  REAL c1 = 2.0*M_PI;
  REAL c2 = 6.0*M_PI;
  REAL p = 13; // must be odd(?) and allows up to p-2 order MG
        *U    =                                                       pow(sin(c1*x),p  )*pow(sin(c1*y),p)*pow(sin(c1*z),p);
        *Ux   =                                        c1*p*cos(c1*x)*pow(sin(c1*x),p-1)*pow(sin(c1*y),p)*pow(sin(c1*z),p);
        *Uy   =                                        c1*p*cos(c1*y)*pow(sin(c1*y),p-1)*pow(sin(c1*x),p)*pow(sin(c1*z),p);
        *Uz   =                                        c1*p*cos(c1*z)*pow(sin(c1*z),p-1)*pow(sin(c1*x),p)*pow(sin(c1*y),p);
        *Uxx  = c1*c1*p*( (p-1)*pow(sin(c1*x),p-2)*pow(cos(c1*x),2) - pow(sin(c1*x),p) )*pow(sin(c1*y),p)*pow(sin(c1*z),p);
        *Uyy  = c1*c1*p*( (p-1)*pow(sin(c1*y),p-2)*pow(cos(c1*y),2) - pow(sin(c1*y),p) )*pow(sin(c1*x),p)*pow(sin(c1*z),p);
        *Uzz  = c1*c1*p*( (p-1)*pow(sin(c1*z),p-2)*pow(cos(c1*z),2) - pow(sin(c1*z),p) )*pow(sin(c1*x),p)*pow(sin(c1*y),p);

        *U   +=                                                       pow(sin(c2*x),p  )*pow(sin(c2*y),p)*pow(sin(c2*z),p);
        *Ux  +=                                        c2*p*cos(c2*x)*pow(sin(c2*x),p-1)*pow(sin(c2*y),p)*pow(sin(c2*z),p);
        *Uy  +=                                        c2*p*cos(c2*y)*pow(sin(c2*y),p-1)*pow(sin(c2*x),p)*pow(sin(c2*z),p);
        *Uz  +=                                        c2*p*cos(c2*z)*pow(sin(c2*z),p-1)*pow(sin(c2*x),p)*pow(sin(c2*y),p);
        *Uxx += c2*c2*p*( (p-1)*pow(sin(c2*x),p-2)*pow(cos(c2*x),2) - pow(sin(c2*x),p) )*pow(sin(c2*y),p)*pow(sin(c2*z),p);
        *Uyy += c2*c2*p*( (p-1)*pow(sin(c2*y),p-2)*pow(cos(c2*y),2) - pow(sin(c2*y),p) )*pow(sin(c2*x),p)*pow(sin(c2*z),p);
        *Uzz += c2*c2*p*( (p-1)*pow(sin(c2*z),p-2)*pow(cos(c2*z),2) - pow(sin(c2*z),p) )*pow(sin(c2*x),p)*pow(sin(c2*y),p);
}


//------------------------------------------------------------------------------------------------------------------------------
void initialize_problem(level_type * level, REAL hLevel, REAL a, REAL b){
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
      REAL x = hLevel*( (REAL)(i+level->my_boxes[box].low.i) + 0.5 ); // +0.5 to get to the center of cell
      REAL y = hLevel*( (REAL)(j+level->my_boxes[box].low.j) + 0.5 );
      REAL z = hLevel*( (REAL)(k+level->my_boxes[box].low.k) + 0.5 );
      REAL A,B,Bx,By,Bz,Bi,Bj,Bk;
      REAL U,Ux,Uy,Uz,Uxx,Uyy,Uzz;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      A  = 1.0;
      B  = 1.0;
      Bx = 0.0;
      By = 0.0;
      Bz = 0.0; 
      Bi = 1.0;
      Bj = 1.0;
      Bk = 1.0;
      #ifdef STENCIL_VARIABLE_COEFFICIENT // variable coefficient problem...
      evaluateBeta(x-hLevel*0.5,y           ,z           ,&Bi,&Bx,&By,&Bz); // face-centered value of Beta for beta_i
      evaluateBeta(x           ,y-hLevel*0.5,z           ,&Bj,&Bx,&By,&Bz); // face-centered value of Beta for beta_j
      evaluateBeta(x           ,y           ,z-hLevel*0.5,&Bk,&Bx,&By,&Bz); // face-centered value of Beta for beta_k
      evaluateBeta(x           ,y           ,z           ,&B ,&Bx,&By,&Bz); // cell-centered value of Beta
      #endif
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      evaluateU(x,y,z,&U,&Ux,&Uy,&Uz,&Uxx,&Uyy,&Uzz, (level->boundary_condition.type == BC_PERIODIC) );
      REAL F = a*A*U - b*( (Bx*Ux + By*Uy + Bz*Uz)  +  B*(Uxx + Uyy + Uzz) );
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
