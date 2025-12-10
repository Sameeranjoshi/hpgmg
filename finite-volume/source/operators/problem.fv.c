//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846 // in case math.h doesn't define it
#endif
REAL evaluateBeta(REAL x, REAL y, REAL z, REAL h, int add_Bxx, int add_Byy, int add_Bzz){
  REAL b = 0.25;
  REAL a = 2.0*M_PI; // one period on [0,1]^3

  REAL B    = 1.0 + b*sin(a*x)*sin(a*y)*sin(a*z);
//REAL Bx   =     a*b*cos(a*x)*sin(a*y)*sin(a*z);
//REAL By   =     a*b*sin(a*x)*cos(a*y)*sin(a*z);
//REAL Bz   =     a*b*sin(a*x)*sin(a*y)*cos(a*z);
  REAL Bxx  =  -a*a*b*sin(a*x)*sin(a*y)*sin(a*z);
  REAL Byy  =  -a*a*b*sin(a*x)*sin(a*y)*sin(a*z);
  REAL Bzz   = -a*a*b*sin(a*x)*sin(a*y)*sin(a*z);

  // 4th order correction to approximate the conversion of cell-centered values to cell-averaged...
  if(add_Bxx)B+=(h*h/24.0)*Bxx;
  if(add_Byy)B+=(h*h/24.0)*Byy;
  if(add_Bzz)B+=(h*h/24.0)*Bzz;
  return(B);
}


//------------------------------------------------------------------------------------------------------------------------------
REAL evaluateF(REAL x, REAL y, REAL z, REAL h, int add_Fxx, int add_Fyy, int add_Fzz){
  #if 0 // harder problem... not sure I manually differentiated this right...
  // 8 'poles', one per octant
  REAL cx = 0.75;
  REAL cy = 0.75;
  REAL cz = 0.75,sign = 1.0;
  if(x<0.5){cx = 0.25;sign*=-1.0;}
  if(y<0.5){cy = 0.25;sign*=-1.0;}
  if(z<0.5){cz = 0.25;sign*=-1.0;}

  REAL r0  = 0.1;
  REAL a   = M_PI/2/r0;
  REAL r   =  pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) ,  0.5); // euclidean distance
  REAL rx  =  pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5)*(x-cx); // dr/dx
  REAL ry  =  pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5)*(y-cy);
  REAL rz  =  pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5)*(z-cz);
  REAL rxx = -pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -1.5)*(x-cx)*(x-cx) + pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5); // d2r/dx2
  REAL ryy = -pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -1.5)*(y-cy)*(y-cy) + pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5);
  REAL rzz = -pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -1.5)*(z-cz)*(z-cz) + pow( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5);

  REAL p   = 6.0;
  REAL F   = sign*(        pow(cos(a*r),p  )    );
  REAL Fx  = sign*(   -a*p*pow(cos(a*r),p-1)*sin(a*r)*rx );
  REAL Fy  = sign*(   -a*p*pow(cos(a*r),p-1)*sin(a*r)*ry );
  REAL Fz  = sign*(   -a*p*pow(cos(a*r),p-1)*sin(a*r)*rz );
  REAL Fxx = sign*( -a*a*p*pow(cos(a*r),p  )*rx*rx  +  a*a*p*(p-1)*pow(cos(a*r),p-2)*pow(sin(a*r),2)*rx*rx  -  a*p*pow(cos(a*r),p-1)*sin(a*r)*rxx );
  REAL Fyy = sign*( -a*a*p*pow(cos(a*r),p  )*ry*ry  +  a*a*p*(p-1)*pow(cos(a*r),p-2)*pow(sin(a*r),2)*ry*ry  -  a*p*pow(cos(a*r),p-1)*sin(a*r)*ryy );
  REAL Fzz = sign*( -a*a*p*pow(cos(a*r),p  )*rz*rz  +  a*a*p*(p-1)*pow(cos(a*r),p-2)*pow(sin(a*r),2)*rz*rz  -  a*p*pow(cos(a*r),p-1)*sin(a*r)*rzz );

  if(r>=r0){
    F   = 0.0;
    Fx  = 0.0;
    Fy  = 0.0;
    Fz  = 0.0;
    Fxx = 0.0;
    Fyy = 0.0;
    Fzz = 0.0;
  }
  #else
  REAL a = 2.0*M_PI;
  REAL p = 1.0;
  REAL F   =        pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  );
//REAL Fx  =    a*p*pow(sin(a*x),p-1)*pow(sin(a*y),p  )*pow(sin(a*z),p  )*cos(a*x);
//REAL Fy  =    a*p*pow(sin(a*x),p  )*pow(sin(a*y),p-1)*pow(sin(a*z),p  )*cos(a*y);
//REAL Fz  =    a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p-1)*cos(a*z);
  REAL Fxx = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p-2)*pow(sin(a*y),p  )*pow(sin(a*z),p  )*pow(cos(a*x),2);
  REAL Fyy = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p  )*pow(sin(a*y),p-2)*pow(sin(a*z),p  )*pow(cos(a*y),2);
  REAL Fzz = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p-2)*pow(cos(a*z),2);
  #endif

  // 4th order correction to approximate the conversion of cell-centered values to cell-averaged...
  if(add_Fxx)F+=(h*h/24.0)*Fxx;
  if(add_Fyy)F+=(h*h/24.0)*Fyy;
  if(add_Fzz)F+=(h*h/24.0)*Fzz;

  return(F);
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
      REAL A,Bi,Bj,Bk;
      //REAL A,B,Bx,By,Bz,Bi,Bj,Bk;
      //REAL U,Ux,Uy,Uz,Uxx,Uyy,Uzz;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      A  = 1.0;
      Bi = 1.0;
      Bj = 1.0;
      Bk = 1.0;
      #ifdef STENCIL_VARIABLE_COEFFICIENT // variable coefficient problem...
      Bi=evaluateBeta(x-hLevel*0.5,y           ,z           ,hLevel,0,1,1); // face-centered value of Beta for beta_i
      Bj=evaluateBeta(x           ,y-hLevel*0.5,z           ,hLevel,1,0,1); // face-centered value of Beta for beta_j
      Bk=evaluateBeta(x           ,y           ,z-hLevel*0.5,hLevel,1,1,0); // face-centered value of Beta for beta_k
      #endif
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      REAL F=evaluateF(x,y,z,hLevel,1,1,1);
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      level->my_boxes[box].vectors[VECTOR_BETA_I][ijk] = Bi;
      level->my_boxes[box].vectors[VECTOR_BETA_J][ijk] = Bj;
      level->my_boxes[box].vectors[VECTOR_BETA_K][ijk] = Bk;
      level->my_boxes[box].vectors[VECTOR_ALPHA ][ijk] = A;
      level->my_boxes[box].vectors[VECTOR_UTRUE ][ijk] = 0.0;
      level->my_boxes[box].vectors[VECTOR_F     ][ijk] = F;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    }}}
  }

}
//------------------------------------------------------------------------------------------------------------------------------
