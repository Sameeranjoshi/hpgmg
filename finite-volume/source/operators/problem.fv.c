//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846f // in case math.h doesn't define it
#endif
float evaluateBeta(float x, float y, float z, float h, int add_Bxx, int add_Byy, int add_Bzz){
  float b = 0.25f;
  float a = 2.0f*M_PI; // one period on [0,1]^3

  float B    = 1.0f + b*sinf(a*x)*sinf(a*y)*sinf(a*z);
//float Bx   =     a*b*cosf(a*x)*sinf(a*y)*sinf(a*z);
//float By   =     a*b*sinf(a*x)*cosf(a*y)*sinf(a*z);
//float Bz   =     a*b*sinf(a*x)*sinf(a*y)*cosf(a*z);
  float Bxx  =  -a*a*b*sinf(a*x)*sinf(a*y)*sinf(a*z);
  float Byy  =  -a*a*b*sinf(a*x)*sinf(a*y)*sinf(a*z);
  float Bzz   = -a*a*b*sinf(a*x)*sinf(a*y)*sinf(a*z);

  // 4th order correction to approximate the conversion of cell-centered values to cell-averaged...
  if(add_Bxx)B+=(h*h/24.0f)*Bxx;
  if(add_Byy)B+=(h*h/24.0f)*Byy;
  if(add_Bzz)B+=(h*h/24.0f)*Bzz;
  return(B);
}


//------------------------------------------------------------------------------------------------------------------------------
float evaluateF(float x, float y, float z, float h, int add_Fxx, int add_Fyy, int add_Fzz){
  #if 0 // harder problem... not sure I manually differentiated this right...
  // 8 'poles', one per octant
  float    cx = 0.75f;
  float    cy = 0.75f;
  float    cz = 0.75f,sign = 1.0f;
  if(x<0.5f){cx = 0.25f;sign*=-1.0f;}
  if(y<0.5f){cy = 0.25f;sign*=-1.0f;}
  if(z<0.5f){cz = 0.25f;sign*=-1.0f;}

  float r0  = 0.1f;
  float a   = M_PI/2/r0;
  float r   =  powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) ,  0.5f); // euclidean distance
  float rx  =  powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5f)*(x-cx); // dr/dx
  float ry  =  powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5f)*(y-cy);
  float rz  =  powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5f)*(z-cz);
  float rxx = -powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -1.5f)*(x-cx)*(x-cx) + powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5f); // d2r/dx2
  float ryy = -powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -1.5f)*(y-cy)*(y-cy) + powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5f);
  float rzz = -powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -1.5f)*(z-cz)*(z-cz) + powf( (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz) , -0.5f);

  float p   = 6.0f;
  float F   = sign*(        powf(cosf(a*r),p  )    );
  float Fx  = sign*(   -a*p*powf(cosf(a*r),p-1)*sinf(a*r)*rx );
  float Fy  = sign*(   -a*p*powf(cosf(a*r),p-1)*sinf(a*r)*ry );
  float Fz  = sign*(   -a*p*powf(cosf(a*r),p-1)*sinf(a*r)*rz );
  float Fxx = sign*( -a*a*p*powf(cosf(a*r),p  )*rx*rx  +  a*a*p*(p-1)*powf(cosf(a*r),p-2)*powf(sinf(a*r),2)*rx*rx  -  a*p*powf(cosf(a*r),p-1)*sinf(a*r)*rxx );
  float Fyy = sign*( -a*a*p*powf(cosf(a*r),p  )*ry*ry  +  a*a*p*(p-1)*powf(cosf(a*r),p-2)*powf(sinf(a*r),2)*ry*ry  -  a*p*powf(cosf(a*r),p-1)*sinf(a*r)*ryy );
  float Fzz = sign*( -a*a*p*powf(cosf(a*r),p  )*rz*rz  +  a*a*p*(p-1)*powf(cosf(a*r),p-2)*powf(sinf(a*r),2)*rz*rz  -  a*p*powf(cosf(a*r),p-1)*sinf(a*r)*rzz );

  if(r>=r0){
    F   = 0.0f;
    Fx  = 0.0f;
    Fy  = 0.0f;
    Fz  = 0.0f;
    Fxx = 0.0f;
    Fyy = 0.0f;
    Fzz = 0.0f;
  }
  #else
  float a = 2.0f*M_PI;
  float p = 1.0f;
  float F   =        powf(sinf(a*x),p  )*powf(sinf(a*y),p  )*powf(sinf(a*z),p  );
//float Fx  =    a*p*powf(sinf(a*x),p-1)*powf(sinf(a*y),p  )*powf(sinf(a*z),p  )*cosf(a*x);
//float Fy  =    a*p*powf(sinf(a*x),p  )*powf(sinf(a*y),p-1)*powf(sinf(a*z),p  )*cosf(a*y);
//float Fz  =    a*p*powf(sinf(a*x),p  )*powf(sinf(a*y),p  )*powf(sinf(a*z),p-1)*cosf(a*z);
  float Fxx = -a*a*p*powf(sinf(a*x),p  )*powf(sinf(a*y),p  )*powf(sinf(a*z),p  )  +  a*a*p*(p-1)*powf(sinf(a*x),p-2)*powf(sinf(a*y),p  )*powf(sinf(a*z),p  )*powf(cosf(a*x),2);
  float Fyy = -a*a*p*powf(sinf(a*x),p  )*powf(sinf(a*y),p  )*powf(sinf(a*z),p  )  +  a*a*p*(p-1)*powf(sinf(a*x),p  )*powf(sinf(a*y),p-2)*powf(sinf(a*z),p  )*powf(cosf(a*y),2);
  float Fzz = -a*a*p*powf(sinf(a*x),p  )*powf(sinf(a*y),p  )*powf(sinf(a*z),p  )  +  a*a*p*(p-1)*powf(sinf(a*x),p  )*powf(sinf(a*y),p  )*powf(sinf(a*z),p-2)*powf(cosf(a*z),2);
  #endif

  // 4th order correction to approximate the conversion of cell-centered values to cell-averaged...
  if(add_Fxx)F+=(h*h/24.0f)*Fxx;
  if(add_Fyy)F+=(h*h/24.0f)*Fyy;
  if(add_Fzz)F+=(h*h/24.0f)*Fzz;

  return(F);
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
      float A,Bi,Bj,Bk;
      //float A,B,Bx,By,Bz,Bi,Bj,Bk;
      //float U,Ux,Uy,Uz,Uxx,Uyy,Uzz;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      A  = 1.0f;
      Bi = 1.0f;
      Bj = 1.0f;
      Bk = 1.0f;
      #ifdef STENCIL_VARIABLE_COEFFICIENT // variable coefficient problem...
      Bi=evaluateBeta(x-hLevel*0.5f,y           ,z           ,hLevel,0,1,1); // face-centered value of Beta for beta_i
      Bj=evaluateBeta(x           ,y-hLevel*0.5f,z           ,hLevel,1,0,1); // face-centered value of Beta for beta_j
      Bk=evaluateBeta(x           ,y           ,z-hLevel*0.5f,hLevel,1,1,0); // face-centered value of Beta for beta_k
      #endif
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      float F=evaluateF(x,y,z,hLevel,1,1,1);
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      level->my_boxes[box].vectors[VECTOR_BETA_I][ijk] = Bi;
      level->my_boxes[box].vectors[VECTOR_BETA_J][ijk] = Bj;
      level->my_boxes[box].vectors[VECTOR_BETA_K][ijk] = Bk;
      level->my_boxes[box].vectors[VECTOR_ALPHA ][ijk] = A;
      level->my_boxes[box].vectors[VECTOR_UTRUE ][ijk] = 0.0f;
      level->my_boxes[box].vectors[VECTOR_F     ][ijk] = F;
      //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    }}}
  }

}
//------------------------------------------------------------------------------------------------------------------------------
