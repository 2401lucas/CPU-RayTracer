#pragma once
#include "render/Camera.h"
#include "render/Scene.h"

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Render(const char* filename, const uint32_t width, const uint32_t height,
              const uint32_t samples);

 private:
  void IntersectScene(Rays* input_rays);
  int CountActiveMasks(Rays* input_rays);
  void CompactRays(Rays* rays, float* image);

  Scene* _scene;
};