#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include "../render/Ray.h"

class Camera {
 public:
  Camera(int width, int height, float fov_deg, glm::vec3 pos, glm::vec3 look_at,
         float aperture_size = 0.0f, float focus_dist = 1.0f);

  void Update(int width, int height, float fov_deg, glm::vec3 pos,
              glm::vec3 look_at, float aperture_size = 0.0f,
              float focus_dist = 1.0f);

  void GenerateRays(Rays& rays, int samples_per_pixel);

 private:
  glm::vec3 up = glm::vec3(0, 1, 0);

  int _width;
  int _height;
  float _fov;
  float _aspect_ratio;
  bool use_dof;

  glm::vec3 _position;
  glm::vec3 _forward, _right, _up;
  float _aperture, _focus_dist, _focal_length = 1.0f;  // DOF
  float viewport_width;
  float viewport_height;

  glm::vec3 _viewport_v, _viewport_h, _viewport_ll;
};