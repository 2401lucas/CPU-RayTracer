#pragma once
#include <immintrin.h>

#include <cstdint>

class Rays {
 public:
  Rays(int max_rays);
  ~Rays();

  int capacity;
  int count;
  int *mask;
  int *pixel_index;
  float *origin_x;
  float *origin_y;
  float *origin_z;
  float *direction_x;
  float *direction_y;
  float *direction_z;
  float *t_min;
  float *t_max;
  float *color_r;
  float *color_g;
  float *color_b;
};