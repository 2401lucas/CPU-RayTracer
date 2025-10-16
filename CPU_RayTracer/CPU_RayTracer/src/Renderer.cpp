#include "Renderer.h"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "../includes/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../includes/stb_image_write.h"

Renderer::Renderer() {
  _scene = new Scene(100);
  _scene->count = 100;

  for (size_t i = 0; i < 100; i++) {
    _scene->center_x[i] = 0.f;
    _scene->center_y[i] = 0.f;
    _scene->center_z[i] = 2.f;
    _scene->radius[i] = 1.1f;
    _scene->mat_r[i] = 1.f;
    _scene->mat_g[i] = 0.f;
    _scene->mat_b[i] = 0.f;
  }
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
  Camera camera(padded_width, height, 90.0f, glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f), 0.0f, 1.0f);

  Rays* rays = new Rays(padded_ray_count);
  camera.GenerateRays(*rays, samples);

  while (rays->count > 0) {
    IntersectScene(rays);

    int active = CountActiveMasks(rays);
    if (active < rays->count * 0.5f) {
      CompactRays(rays, image);
    }
  }

  for (int i = 0; i < total_pixels; i++) {
    final_image[i] = (uint8_t)(std::min(1.0f, image[i]) * 255.0f);
  }

  stbi_write_png(filename, padded_width, height, CHANNEL_NUM, final_image,
                 padded_width * CHANNEL_NUM);

   delete rays;
   delete[] image;
   delete[] final_image;
}

void Renderer::IntersectScene(Rays* input_rays) {
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

    for (int j = 0; j < _scene->count; ++j) {
      __m256 cx = _mm256_set1_ps(_scene->center_x[j]);
      __m256 cy = _mm256_set1_ps(_scene->center_y[j]);
      __m256 cz = _mm256_set1_ps(_scene->center_z[j]);
      __m256 r = _mm256_set1_ps(_scene->radius[j]);

      // oc = origin - center
      __m256 ocx = _mm256_sub_ps(ox, cx);
      __m256 ocy = _mm256_sub_ps(oy, cy);
      __m256 ocz = _mm256_sub_ps(oz, cz);

      // quadratic coefficients (ray dir normalized assumption : a = 1)
      __m256 b = _mm256_add_ps(
          _mm256_mul_ps(ocx, dx),
          _mm256_add_ps(_mm256_mul_ps(ocy, dy), _mm256_mul_ps(ocz, dz)));

      // c = dot(oc,oc) - r^2
      __m256 c = _mm256_sub_ps(
          _mm256_add_ps(
              _mm256_mul_ps(ocx, ocx),
              _mm256_add_ps(_mm256_mul_ps(ocy, ocy), _mm256_mul_ps(ocz, ocz))),
          _mm256_mul_ps(r, r));

      // discriminant = b*b - c
      __m256 b2 = _mm256_mul_ps(b, b);
      __m256 disc = _mm256_sub_ps(b2, c);

      // disc > 0 ?
      __m256 discMask = _mm256_cmp_ps(disc, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
      // no lane has intersection with this sphere
      if (_mm256_movemask_ps(discMask) == 0) continue;

      // sqrt(disc)
      __m256 sqd = _mm256_sqrt_ps(disc);
      __m256 t0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), b),
                                sqd);  // -b - sqrt
      __m256 t1 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), b),
                                sqd);  // -b + sqrt

      // choose nearer positive root: prefer t0 if > t_min, else t1
      __m256 t0_gt_t_min = _mm256_cmp_ps(t0, t_min, _CMP_GT_OQ);
      __m256 t_candidate =
          _mm256_blendv_ps(t1, t0, t0_gt_t_min);  // if t0>t_min use t0 else t1

      // valid = discMask && (t_candidate > t_min) && (t_candidate < t_best)
      __m256 gt_tMin = _mm256_cmp_ps(t_candidate, t_min, _CMP_GT_OQ);
      __m256 lt_tBest = _mm256_cmp_ps(t_candidate, t_best, _CMP_LT_OQ);
      __m256 valid = _mm256_and_ps(discMask, _mm256_and_ps(gt_tMin, lt_tBest));

      // update t_best, and store sphere color/center for lanes where valid
      t_best = _mm256_blendv_ps(t_best, t_candidate,
                                valid);  // if valid, t_best = t_candidate

      __m256 sr = _mm256_set1_ps(_scene->mat_r[j]);
      __m256 sg = _mm256_set1_ps(_scene->mat_g[j]);
      __m256 sb = _mm256_set1_ps(_scene->mat_b[j]);

      best_r = _mm256_blendv_ps(best_r, sr, valid);
      best_g = _mm256_blendv_ps(best_g, sg, valid);
      best_b = _mm256_blendv_ps(best_b, sb, valid);

      hit_center_x = _mm256_blendv_ps(hit_center_x, cx, valid);
      hit_center_y = _mm256_blendv_ps(hit_center_y, cy, valid);
      hit_center_z = _mm256_blendv_ps(hit_center_z, cz, valid);
    }

    // Determine lanes that hit something: hit = t_best < original t_max
    __m256 hit_mask = _mm256_cmp_ps(t_best, t_max, _CMP_LT_OQ);

    // compute hit point p = o + d * t_best
    __m256 px = _mm256_add_ps(ox, _mm256_mul_ps(dx, t_best));
    __m256 py = _mm256_add_ps(oy, _mm256_mul_ps(dy, t_best));
    __m256 pz = _mm256_add_ps(oz, _mm256_mul_ps(dz, t_best));

    // compute normal = normalize(p - hitCenter)
    __m256 nx = _mm256_sub_ps(px, hit_center_x);
    __m256 ny = _mm256_sub_ps(py, hit_center_y);
    __m256 nz = _mm256_sub_ps(pz, hit_center_z);

    // length and normalize
    __m256 len2 = _mm256_add_ps(
        _mm256_mul_ps(nx, nx),
        _mm256_add_ps(_mm256_mul_ps(ny, ny), _mm256_mul_ps(nz, nz)));
    __m256 inv_length = _mm256_rsqrt_ps(len2);  // approximate inverse sqrt
    nx = _mm256_mul_ps(nx, inv_length);
    ny = _mm256_mul_ps(ny, inv_length);
    nz = _mm256_mul_ps(nz, inv_length);

    __m256 out_r = best_r;
    __m256 out_g = best_g;
    __m256 out_b = best_b;

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

    // store colors back to ray arrays (as floats)
    _mm256_storeu_ps(&input_rays->color_r[i], out_r);
    _mm256_storeu_ps(&input_rays->color_g[i], out_g);
    _mm256_storeu_ps(&input_rays->color_b[i], out_b);

    // mark these lanes as terminated (mask = 0) since this is single-hit tracer
    __m256i zeroi = _mm256_setzero_si256();
    _mm256_store_si256((__m256i*)&input_rays->mask[i], zeroi);
  }
}

int Renderer::CountActiveMasks(Rays* input_rays) {
  int total = 0;

  for (int i = 0; i < input_rays->count; i += 8) {
    __m256i mask_vec = _mm256_load_si256((__m256i*)&input_rays->mask[i]);

    int bitmask = _mm256_movemask_ps(_mm256_castsi256_ps(mask_vec));
    total += __popcnt(bitmask);  // Count 1-bits
  }

  return total;
}

void Renderer::CompactRays(Rays* rays, float* image) {
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
      image[pixel + 0] += rays->color_r[i];
      image[pixel + 1] += rays->color_g[i];
      image[pixel + 2] += rays->color_b[i];
    }
  }
  rays->count = writeIdx;
}