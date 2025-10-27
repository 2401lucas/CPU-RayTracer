#include "Camera.h"

#include <algorithm>
#include <glm/gtc/constants.hpp>
#include <iostream>

#include "../includes/simd_helper.h"

Camera::Camera(int width, int height, float fov_deg, glm::vec3 pos,
               glm::vec3 look_at, float aperture_size, float focus_dist) {
  Update(width, height, fov_deg, pos, look_at, aperture_size, focus_dist);
}

void Camera::Update(int width, int height, float fov_deg, glm::vec3 pos,
                    glm::vec3 look_at, float aperture_size, float focus_dist) {
  _width = width;
  _height = height;
  _fov = fov_deg;
  _position = pos;
  _focus_dist = focus_dist;
  _aperture = aperture_size;
  use_dof = aperture_size > 0.0f;
  _forward = glm::normalize(look_at - pos);
  _right = glm::normalize(glm::cross(_forward, up));
  _up = glm::cross(_right, _forward);

  float theta = _fov * (glm::pi<float>() / 180.0f);
  _aspect_ratio = (float)_width / _height;
  viewport_height = 2.0f * std::tan(theta / 2.0f) * _focal_length;
  viewport_width = viewport_height * _aspect_ratio;

  _viewport_v = viewport_height * _up;
  _viewport_h = viewport_width * _right;
  _viewport_ll = pos + _focal_length * _forward - _viewport_h / glm::vec3(2.f) -
                 _viewport_v / glm::vec3(2.f);
}

void Camera::GenerateRays(Rays& rays, int samples_per_pixel = 1) {
  rays.count = _width * _height * samples_per_pixel;
  __m256 origin_x_vec = _mm256_set1_ps(_position.x);
  __m256 origin_y_vec = _mm256_set1_ps(_position.y);
  __m256 origin_z_vec = _mm256_set1_ps(_position.z);

  __m256 lower_left_x_vec = _mm256_set1_ps(_viewport_ll.x);
  __m256 lower_left_y_vec = _mm256_set1_ps(_viewport_ll.y);
  __m256 lower_left_z_vec = _mm256_set1_ps(_viewport_ll.z);

  __m256 horiz_x_vec = _mm256_set1_ps(_viewport_h.x);
  __m256 horiz_y_vec = _mm256_set1_ps(_viewport_h.y);
  __m256 horiz_z_vec = _mm256_set1_ps(_viewport_h.z);

  __m256 vert_x_vec = _mm256_set1_ps(_viewport_v.x);
  __m256 vert_y_vec = _mm256_set1_ps(_viewport_v.y);
  __m256 vert_z_vec = _mm256_set1_ps(_viewport_v.z);

  __m256 width_vec = _mm256_set1_ps(float(_width));
  __m256 height_vec = _mm256_set1_ps(float(_height));

  __m256 tmin_vec = _mm256_set1_ps(0.001f);
  __m256 tmax_vec = _mm256_set1_ps(FLT_MAX);

  __m256 zero_vec = _mm256_setzero_ps();

  // For jittered sampling
  __m256 inv_spp = _mm256_set1_ps(1.0f / float(samples_per_pixel));

  // Dof vectors
  __m256 aperture_vec = _mm256_set1_ps(_aperture * 0.5f);
  __m256 focus_dist_vec = _mm256_set1_ps(_focus_dist);

  int idx = 0;

  __m256i rand_state = _mm256_set_epi32(12345678, 23456789, 34567890, 45678901,
                                        56789012, 67890123, 78901234, 89012345);
  for (int sample = 0; sample < samples_per_pixel; sample++) {
    for (int y = 0; y < _height; y++) {
      for (int x = 0; x < _width; x += 8) {
        __m256 x_coords = _mm256_set_ps(
            float(x + 7), float(x + 6), float(x + 5), float(x + 4),
            float(x + 3), float(x + 2), float(x + 1), float(x));
        __m256 y_coords = _mm256_set1_ps(float(y));

        // AA (stratified sampling)
        if (samples_per_pixel > 1) {
          __m256 jitter_x = simd::Random(rand_state);
          __m256 jitter_y = simd::Random(rand_state);
          x_coords = _mm256_add_ps(x_coords, jitter_x);
          y_coords = _mm256_add_ps(y_coords, jitter_y);
        } else {
          x_coords = _mm256_add_ps(x_coords, _mm256_set1_ps(0.5f));
          y_coords = _mm256_add_ps(y_coords, _mm256_set1_ps(0.5f));
        }

        __m256 u = _mm256_div_ps(x_coords, width_vec);
        __m256 v = _mm256_div_ps(y_coords, height_vec);

        __m256 dir_x = _mm256_add_ps(
            lower_left_x_vec, _mm256_add_ps(_mm256_mul_ps(u, horiz_x_vec),
                                            _mm256_mul_ps(v, vert_x_vec)));
        dir_x = _mm256_sub_ps(dir_x, origin_x_vec);

        __m256 dir_y = _mm256_add_ps(
            lower_left_y_vec, _mm256_add_ps(_mm256_mul_ps(u, horiz_y_vec),
                                            _mm256_mul_ps(v, vert_y_vec)));
        dir_y = _mm256_sub_ps(dir_y, origin_y_vec);

        __m256 dir_z = _mm256_add_ps(
            lower_left_z_vec, _mm256_add_ps(_mm256_mul_ps(u, horiz_z_vec),
                                            _mm256_mul_ps(v, vert_z_vec)));
        dir_z = _mm256_sub_ps(dir_z, origin_z_vec);

        // Apply dof if enabled
        __m256 ray_origin_x = origin_x_vec;
        __m256 ray_origin_y = origin_y_vec;
        __m256 ray_origin_z = origin_z_vec;

        if (use_dof) {
          // Generate random point on lens (unit disk scaled by aperture)
          __m256 disk_x, disk_y;
          simd::RandomInUnitDisc(rand_state, disk_x, disk_y);
          disk_x = _mm256_mul_ps(disk_x, aperture_vec);
          disk_y = _mm256_mul_ps(disk_y, aperture_vec);

          // Offset origin by lens sample
          __m256 offset_x =
              _mm256_add_ps(_mm256_mul_ps(disk_x, _mm256_set1_ps(_right.x)),
                            _mm256_mul_ps(disk_y, _mm256_set1_ps(_up.x)));
          __m256 offset_y =
              _mm256_add_ps(_mm256_mul_ps(disk_x, _mm256_set1_ps(_right.y)),
                            _mm256_mul_ps(disk_y, _mm256_set1_ps(_up.y)));
          __m256 offset_z =
              _mm256_add_ps(_mm256_mul_ps(disk_x, _mm256_set1_ps(_right.z)),
                            _mm256_mul_ps(disk_y, _mm256_set1_ps(_up.z)));

          ray_origin_x = _mm256_add_ps(ray_origin_x, offset_x);
          ray_origin_y = _mm256_add_ps(ray_origin_y, offset_y);
          ray_origin_z = _mm256_add_ps(ray_origin_z, offset_z);

          // Adjust direction to focus point
          __m256 focus_x =
              _mm256_add_ps(origin_x_vec, _mm256_mul_ps(dir_x, focus_dist_vec));
          __m256 focus_y =
              _mm256_add_ps(origin_y_vec, _mm256_mul_ps(dir_y, focus_dist_vec));
          __m256 focus_z =
              _mm256_add_ps(origin_z_vec, _mm256_mul_ps(dir_z, focus_dist_vec));

          dir_x = _mm256_sub_ps(focus_x, ray_origin_x);
          dir_y = _mm256_sub_ps(focus_y, ray_origin_y);
          dir_z = _mm256_sub_ps(focus_z, ray_origin_z);
        }

        simd::Normalize(dir_x, dir_y, dir_z);

        _mm256_store_ps(&rays.origin_x[idx], ray_origin_x);
        _mm256_store_ps(&rays.origin_y[idx], ray_origin_y);
        _mm256_store_ps(&rays.origin_z[idx], ray_origin_z);

        _mm256_store_ps(&rays.direction_x[idx], dir_x);
        _mm256_store_ps(&rays.direction_y[idx], dir_y);
        _mm256_store_ps(&rays.direction_z[idx], dir_z);

        _mm256_store_ps(&rays.t_min[idx], tmin_vec);
        _mm256_store_ps(&rays.t_max[idx], tmax_vec);

        _mm256_store_ps(&rays.color_r[idx], zero_vec);
        _mm256_store_ps(&rays.color_g[idx], zero_vec);
        _mm256_store_ps(&rays.color_b[idx], zero_vec);

        __m256i pixel_indices = _mm256_set_epi32(
            y * _width + (x + 7), y * _width + (x + 6), y * _width + (x + 5),
            y * _width + (x + 4), y * _width + (x + 3), y * _width + (x + 2),
            y * _width + (x + 1), y * _width + (x + 0));
        _mm256_store_si256((__m256i*)&rays.pixel_index[idx], pixel_indices);

        idx += 8;
      }
    }
  }
}
