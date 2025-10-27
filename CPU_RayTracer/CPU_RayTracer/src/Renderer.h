#pragma once
#include "../includes/ThreadPool.h"
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
  void IntersectScene_NoBVH(Rays* rays);
  void IntersectScene_BVH8(Rays* rays);
  void IntersectScene_Mt(Rays* rays);
  void CompactRays(Rays* rays, float* image, float contrib_factor = 1);

  Scene* _scene;
};
