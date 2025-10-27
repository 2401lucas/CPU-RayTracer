#include "Ray.h"

Rays::Rays(int max_rays) : capacity(max_rays), count(0) {
  pixel_index = (int*)_mm_malloc(max_rays * sizeof(int), 32);
  origin_x = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  origin_y = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  origin_z = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  direction_x = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  direction_y = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  direction_z = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  t_min = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  t_max = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  color_r = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  color_g = (float*)_mm_malloc(max_rays * sizeof(float), 32);
  color_b = (float*)_mm_malloc(max_rays * sizeof(float), 32);
}

Rays::~Rays() {
  _mm_free(pixel_index);
  _mm_free(origin_x);
  _mm_free(origin_y);
  _mm_free(origin_z);
  _mm_free(direction_x);
  _mm_free(direction_y);
  _mm_free(direction_z);
  _mm_free(t_min);
  _mm_free(t_max);
  _mm_free(color_r);
  _mm_free(color_g);
  _mm_free(color_b);
}