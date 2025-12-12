//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifndef OPERATORS_H
#define OPERATORS_H
//------------------------------------------------------------------------------------------------------------------------------
#define RESTRICT_CELL   0
#define RESTRICT_FACE_I 1
#define RESTRICT_FACE_J 2
#define RESTRICT_FACE_K 3
//------------------------------------------------------------------------------------------------------------------------------
int stencil_get_radius(); 
int stencil_get_shape();
//------------------------------------------------------------------------------------------------------------------------------
  void                  apply_op(level_type * level, int Ax_id,  int x_id, float a, float b);
  void                  residual(level_type * level, int res_id, int x_id, int rhs_id, float a, float b);
  void                    smooth(level_type * level, int phi_id, int rhs_id, float a, float b);
  void          rebuild_operator(level_type * level, level_type *fromLevel, float a, float b);
  void rebuild_operator_blackbox(level_type * level, float a, float b, int colors_in_each_dim);
//------------------------------------------------------------------------------------------------------------------------------
  void               restriction(level_type * level_c, int id_c, level_type *level_f, int id_f, int restrictionType);
  void      interpolation_vcycle(level_type * level_f, int id_f, float prescale_f, level_type *level_c, int id_c); // interpolation used inside a v-cycle
  void      interpolation_fcycle(level_type * level_f, int id_f, float prescale_f, level_type *level_c, int id_c); // interpolation used in the f-cycle to create a new initial guess for the next finner v-cycle
//------------------------------------------------------------------------------------------------------------------------------
  void         exchange_boundary(level_type * level, int id_a, int shape);
  void              apply_BCs_p1(level_type * level, int x_id, int shape); // piecewise (cell centered) linear
  void              apply_BCs_p2(level_type * level, int x_id, int shape); // piecewise (cell centered) quadratic
  void              apply_BCs_v1(level_type * level, int x_id, int shape); // volumetric linear
  void              apply_BCs_v2(level_type * level, int x_id, int shape); // volumetric quadratic
  void              apply_BCs_v4(level_type * level, int x_id, int shape); // volumetric quartic
  void         extrapolate_betas(level_type * level);
//------------------------------------------------------------------------------------------------------------------------------
float                       dot(level_type * level, int id_a, int id_b);
float                      norm(level_type * level, int id_a);
float                      mean(level_type * level, int id_a);
float                     error(level_type * level, int id_a, int id_b);
  void               add_vectors(level_type * level, int id_c, float scale_a, int id_a, float scale_b, int id_b);
  void             scale_vector( level_type * level, int id_c, float scale_a, int id_a);
  void              zero_vector( level_type * level, int id_a);
  void             shift_vector( level_type * level, int id_c, int id_a, float shift_a);
  void               mul_vectors(level_type * level, int id_c, float scale, int id_a, int id_b);
  void            invert_vector( level_type * level, int id_c, float scale_a, int id_a);
  void              init_vector( level_type * level, int id_a, float scalar);
//------------------------------------------------------------------------------------------------------------------------------
void                color_vector(level_type * level, int id, int colors, int icolor, int jcolor, int kcolor);
void               random_vector(level_type * level, int id);
//------------------------------------------------------------------------------------------------------------------------------
  void        initialize_problem(level_type * level, float hLevel, float a, float b);
  void   initialize_valid_region(level_type * level);
//------------------------------------------------------------------------------------------------------------------------------
#endif
