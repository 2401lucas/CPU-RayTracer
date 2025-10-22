#pragma once
#include "render/Camera.h"
#include "render/Scene.h"

constexpr float epsilon = 0.0001f;
constexpr bool backface_cull = true;

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Render(const char* filename, const uint32_t width, const uint32_t height,
              const uint32_t samples);

 private:
  void IntersectScene_NoBVH(Rays* input_rays);
  int CountActiveMasks(Rays* input_rays);
  void CompactRays(Rays* rays, float* image, float contrib_factor = 1);

  Scene* _scene;
};