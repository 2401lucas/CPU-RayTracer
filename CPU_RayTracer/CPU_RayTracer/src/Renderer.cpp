#include "Renderer.h"

#include <iostream>

#include "../includes/simd_helper.h"
#include "../includes/stb_image_incl.h"

Renderer::Renderer() {
  _scene = new Scene(100, 1000);
  _scene->LoadScene("demo");
}

Renderer::~Renderer() { delete _scene; }

constexpr uint8_t CHANNEL_NUM = 3;
void Renderer::Render(const char* filename, const uint32_t width,
                      const uint32_t height, const uint32_t samples) {
  int padded_width = ((width + 7) / 8) * 8;
  int total_pixels = padded_width * height * CHANNEL_NUM;
  uint8_t* final_image = new uint8_t[total_pixels];
  float* image = new float[total_pixels];
  std::fill(image, image + total_pixels, 0);
  auto padded_ray_count = padded_width * height * samples;
  float r_samples = 1.f / samples;
  Camera camera(padded_width, height, 90.0f, glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f), 0.f, 1.0f);

  Rays* rays = new Rays(padded_ray_count);
  camera.GenerateRays(*rays, samples);

  IntersectScene_NoBVH(rays);
  CompactRays(rays, image, r_samples);

  for (int i = 0; i < total_pixels; i++) {
    final_image[i] = (uint8_t)(std::min(1.0f, image[i]) * 255.0f);
  }

  stbi_write_png(filename, padded_width, height, CHANNEL_NUM, final_image,
                 padded_width * CHANNEL_NUM);

  delete rays;
  delete[] image;
  delete[] final_image;
}

// This is the most basic single hit scene tracing with no lighting
// 8 rays tested per primitive
// Will this supprot Bounces?
// I know I will support bounces but this is already noticably slow with a
// limited scene. A BVH is a really obvious optimization and while I would like
// to have a comparison with bounces between a no BVH & BVH, I think time could
// be better spent elsewhere.
void Renderer::IntersectScene_NoBVH(Rays* input_rays) {
  for (int i = 0; i < input_rays->count; i += 8) {
    __m256 ox = _mm256_loadu_ps(&input_rays->origin_x[i]);
    __m256 oy = _mm256_loadu_ps(&input_rays->origin_y[i]);
    __m256 oz = _mm256_loadu_ps(&input_rays->origin_z[i]);

    __m256 dx = _mm256_loadu_ps(&input_rays->direction_x[i]);
    __m256 dy = _mm256_loadu_ps(&input_rays->direction_y[i]);
    __m256 dz = _mm256_loadu_ps(&input_rays->direction_z[i]);

    __m256 t_min = _mm256_loadu_ps(&input_rays->t_min[i]);
    __m256 t_max = _mm256_loadu_ps(&input_rays->t_max[i]);

    __m256 t_best = t_max;
    __m256 best_r = _mm256_setzero_ps();
    __m256 best_g = _mm256_setzero_ps();
    __m256 best_b = _mm256_setzero_ps();
    __m256 hit_center_x = _mm256_setzero_ps();
    __m256 hit_center_y = _mm256_setzero_ps();
    __m256 hit_center_z = _mm256_setzero_ps();

    for (int j = 0; j < _scene->spheres.count; ++j) {
      float sx = _scene->spheres.center_x[j];
      float sy = _scene->spheres.center_y[j];
      float sz = _scene->spheres.center_z[j];

      __m256 out_t;
      __m256 valid = simd::IntersectSphere_NoBVH(ox, oy, oz, dx, dy, dz, sx, sy,
                                                 sz, _scene->spheres.radius[j],
                                                 t_min, t_max, out_t);

      __m256 closer =
          _mm256_and_ps(valid, _mm256_cmp_ps(out_t, t_best, _CMP_LT_OQ));
      t_best = _mm256_blendv_ps(t_best, out_t, closer);

      __m256 r = _mm256_set1_ps(_scene->spheres.mat_r[j]);
      __m256 g = _mm256_set1_ps(_scene->spheres.mat_g[j]);
      __m256 b = _mm256_set1_ps(_scene->spheres.mat_b[j]);

      best_r = _mm256_blendv_ps(best_r, r, closer);
      best_g = _mm256_blendv_ps(best_g, g, closer);
      best_b = _mm256_blendv_ps(best_b, b, closer);

      __m256 cx = _mm256_set1_ps(sx);
      __m256 cy = _mm256_set1_ps(sy);
      __m256 cz = _mm256_set1_ps(sz);

      hit_center_x = _mm256_blendv_ps(hit_center_x, cx, closer);
      hit_center_y = _mm256_blendv_ps(hit_center_y, cy, closer);
      hit_center_z = _mm256_blendv_ps(hit_center_z, cz, closer);
    }

    for (int j = 0; j < _scene->tris.count; ++j) {
      __m256 out_t;
      __m256 out_u, out_v;

      __m256 valid = simd::IntersectTriangle_NoBVH(
          ox, oy, oz, dx, dy, dz, _scene->tris.v0_x[j], _scene->tris.v0_y[j],
          _scene->tris.v0_z[j], _scene->tris.v1_x[j], _scene->tris.v1_y[j],
          _scene->tris.v1_z[j], _scene->tris.v2_x[j], _scene->tris.v2_y[j],
          _scene->tris.v2_z[j], t_min, t_max, out_t, out_u, out_v);

      __m256 closer =
          _mm256_and_ps(valid, _mm256_cmp_ps(out_t, t_best, _CMP_LT_OQ));

      t_best = _mm256_blendv_ps(t_best, out_t, closer);
      __m256 r = _mm256_set1_ps(0.0f);
      __m256 g = _mm256_set1_ps(0.1f);
      __m256 b = _mm256_set1_ps(1.0f);

      best_r = _mm256_blendv_ps(best_r, r, closer);
      best_g = _mm256_blendv_ps(best_g, g, closer);
      best_b = _mm256_blendv_ps(best_b, b, closer);

      // Compute interpolated hit point using barycentric coords
      // w = 1 - u - v
      __m256 w =
          _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(out_u, out_v));

      // Interpolated position
      __m256 hit_px = _mm256_add_ps(
          _mm256_add_ps(
              _mm256_mul_ps(w, _mm256_set1_ps(_scene->tris.v0_x[j])),
              _mm256_mul_ps(out_u, _mm256_set1_ps(_scene->tris.v1_x[j]))),
          _mm256_mul_ps(out_v, _mm256_set1_ps(_scene->tris.v2_x[j])));

      __m256 hit_py = _mm256_add_ps(
          _mm256_add_ps(
              _mm256_mul_ps(w, _mm256_set1_ps(_scene->tris.v0_y[j])),
              _mm256_mul_ps(out_u, _mm256_set1_ps(_scene->tris.v1_y[j]))),
          _mm256_mul_ps(out_v, _mm256_set1_ps(_scene->tris.v2_y[j])));

      __m256 hit_pz = _mm256_add_ps(
          _mm256_add_ps(
              _mm256_mul_ps(w, _mm256_set1_ps(_scene->tris.v0_z[j])),
              _mm256_mul_ps(out_u, _mm256_set1_ps(_scene->tris.v1_z[j]))),
          _mm256_mul_ps(out_v, _mm256_set1_ps(_scene->tris.v2_z[j])));

      // Blend with previous best hit (only replace lanes that are closer)
      hit_center_x = _mm256_blendv_ps(hit_center_x, hit_px, closer);
      hit_center_y = _mm256_blendv_ps(hit_center_y, hit_py, closer);
      hit_center_z = _mm256_blendv_ps(hit_center_z, hit_pz, closer);
    }
    __m256 out_r = best_r;
    __m256 out_g = best_g;
    __m256 out_b = best_b;

    // Determine lanes that hit something: hit = t_best < original t_max
    __m256 hit_mask = _mm256_cmp_ps(t_best, t_max, _CMP_LT_OQ);

    // background color for misses
    __m256 miss_mask = _mm256_cmp_ps(hit_mask, _mm256_setzero_ps(), _CMP_EQ_OQ);
    // sky color based on dir.y
    __m256 t_sky = _mm256_add_ps(_mm256_mul_ps(dy, _mm256_set1_ps(0.5f)),
                                 _mm256_set1_ps(0.5f));
    __m256 sky_r = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(0.7f), _mm256_set1_ps(1.0f)),
        _mm256_mul_ps(_mm256_set1_ps(0.3f), t_sky));  // simple blend
    __m256 sky_g = sky_r;
    __m256 sky_b = _mm256_set1_ps(1.0f);

    out_r = _mm256_blendv_ps(out_r, sky_r, miss_mask);
    out_g = _mm256_blendv_ps(out_g, sky_g, miss_mask);
    out_b = _mm256_blendv_ps(out_b, sky_b, miss_mask);

    // store colors to ray
    _mm256_storeu_ps(&input_rays->color_r[i], out_r);
    _mm256_storeu_ps(&input_rays->color_g[i], out_g);
    _mm256_storeu_ps(&input_rays->color_b[i], out_b);

    // mark these lanes as terminated since this is a single-hit tracer
    // TODO: bounces
    __m256i zeroi = _mm256_setzero_si256();
    _mm256_store_si256((__m256i*)&input_rays->mask[i], zeroi);
  }
}

int Renderer::CountActiveMasks(Rays* input_rays) {
  int total = 0;

  for (int i = 0; i < input_rays->count; i += 8) {
    __m256i mask_vec = _mm256_load_si256((__m256i*)&input_rays->mask[i]);
    total += simd::ActiveMaskCount(_mm256_castsi256_ps(mask_vec));
  }

  return total;
}

void Renderer::CompactRays(Rays* rays, float* image, float contrib_factor) {
  int writeIdx = 0;
  for (int i = 0; i < rays->count; i++) {
    if (rays->mask[i]) {
      rays->pixel_index[writeIdx] = rays->pixel_index[i];
      rays->origin_x[writeIdx] = rays->origin_x[i];
      rays->origin_y[writeIdx] = rays->origin_y[i];
      rays->origin_z[writeIdx] = rays->origin_z[i];
      rays->direction_x[writeIdx] = rays->direction_x[i];
      rays->direction_y[writeIdx] = rays->direction_y[i];
      rays->direction_z[writeIdx] = rays->direction_z[i];
      rays->t_min[writeIdx] = rays->t_min[i];
      rays->t_max[writeIdx] = rays->t_max[i];
      rays->color_r[writeIdx] = rays->color_r[i];
      rays->color_g[writeIdx] = rays->color_g[i];
      rays->color_b[writeIdx] = rays->color_b[i];
      rays->mask[writeIdx] = rays->mask[i];
      writeIdx++;
    } else {
      int pixel = rays->pixel_index[i] * CHANNEL_NUM;
      image[pixel + 0] += rays->color_r[i] * contrib_factor;
      image[pixel + 1] += rays->color_g[i] * contrib_factor;
      image[pixel + 2] += rays->color_b[i] * contrib_factor;
    }
  }
  rays->count = writeIdx;
}