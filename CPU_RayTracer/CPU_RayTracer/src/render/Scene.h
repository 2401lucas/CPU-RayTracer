#pragma once

class Scene {
 public:
  Scene(int max_models) {
    capacity = max_models;
    count = 0;
    center_x = (float*)_mm_malloc(max_models * sizeof(float), 32);
    center_y = (float*)_mm_malloc(max_models * sizeof(float), 32);
    center_z = (float*)_mm_malloc(max_models * sizeof(float), 32);
    radius = (float*)_mm_malloc(max_models * sizeof(float), 32);
    mat_r = (float*)_mm_malloc(max_models * sizeof(float), 32);
    mat_g = (float*)_mm_malloc(max_models * sizeof(float), 32);
    mat_b = (float*)_mm_malloc(max_models * sizeof(float), 32);
  }
  ~Scene() {
    _mm_free(center_x);
    _mm_free(center_y);
    _mm_free(center_z);
    _mm_free(radius);
    _mm_free(mat_r);
    _mm_free(mat_g);
    _mm_free(mat_b);
  }

  void LoadScene(char* filepath) {}
  void LoadModel(char* filepath) {}

  int capacity;
  int count;
  float* center_x;
  float* center_y;
  float* center_z;
  float* radius;
  float* mat_r;
  float* mat_g;
  float* mat_b;
};