#include "Renderer.h"

#include <chrono>
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
  Camera camera(padded_width, height, 90.0f, glm::vec3(2.0f, 0.0f, -4.0f),
                glm::vec3(0.0f, 0.f, 1.0f), 0.0f, 1.0f);

  Rays* rays = new Rays(padded_ray_count);
  camera.GenerateRays(*rays, samples);

  std::cout << "Render Stats: \n Rays: " << padded_ray_count << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  IntersectScene_NoBVH(rays);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Render 1: No BVH\n Took " << elapsed.count() << " seconds."
            << std::endl;

  start = std::chrono::high_resolution_clock::now();
  IntersectScene_BVH8(rays);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Render 2: BVH8\n Took " << elapsed.count() << " seconds."
            << std::endl;

  start = std::chrono::high_resolution_clock::now();
  IntersectScene_Mt(rays);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Render 2: BVH8 + Multithreading\n Took " << elapsed.count()
            << " seconds." << std::endl;

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
void Renderer::IntersectScene_NoBVH(Rays* rays) {
  for (int i = 0; i < rays->count; i += 8) {
    __m256 ox = _mm256_load_ps(&rays->origin_x[i]);
    __m256 oy = _mm256_load_ps(&rays->origin_y[i]);
    __m256 oz = _mm256_load_ps(&rays->origin_z[i]);
    __m256 dx = _mm256_load_ps(&rays->direction_x[i]);
    __m256 dy = _mm256_load_ps(&rays->direction_y[i]);
    __m256 dz = _mm256_load_ps(&rays->direction_z[i]);
    __m256 t_min = _mm256_load_ps(&rays->t_min[i]);
    __m256 t_max = _mm256_load_ps(&rays->t_max[i]);

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
      __m256 valid =
          simd::IntersectSphere(ox, oy, oz, dx, dy, dz, sx, sy, sz,
                                _scene->spheres.radius[j], t_min, t_max, out_t);

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

      __m256 valid = simd::IntersectTriangle(
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
    _mm256_storeu_ps(&rays->color_r[i], out_r);
    _mm256_storeu_ps(&rays->color_g[i], out_g);
    _mm256_storeu_ps(&rays->color_b[i], out_b);
  }
}

// Workload processing, stack of rays that get processed?
// Per ray, if miss -> end
// if hit -> update ray info with pos, dir and contrib, add to stack
// Per ray contrib has influences, 1/ray & ray_bounces_contrib that is reduces
// per bounce, similar to x^n where : x < 1 && x > 0? If bounce facter gets too
// small, end
// I think that max bounces could be the cutoff and it could probably be pretty
// low assuming the contrib factor gradually approaches 0
// By having a ray stack, this allows for proper support for a queue stealing
// multithreading system as well as divergence, IE 1 ray->2 rays
void Renderer::IntersectScene_BVH8(Rays* rays) {
  const int ray_count = rays->count;

  for (int base = 0; base < ray_count; base++) {
    __m256 ox = _mm256_set1_ps(rays->origin_x[base]);
    __m256 oy = _mm256_set1_ps(rays->origin_y[base]);
    __m256 oz = _mm256_set1_ps(rays->origin_z[base]);
    __m256 dx = _mm256_set1_ps(rays->direction_x[base]);
    __m256 dy = _mm256_set1_ps(rays->direction_y[base]);
    __m256 dz = _mm256_set1_ps(rays->direction_z[base]);
    __m256 t_min = _mm256_set1_ps(rays->t_min[base]);
    __m256 t_max = _mm256_set1_ps(rays->t_max[base]);

    __m256 inv_dx = _mm256_rcp_ps(dx);
    __m256 inv_dy = _mm256_rcp_ps(dy);
    __m256 inv_dz = _mm256_rcp_ps(dz);

    float t_best = rays->t_max[base];
    float best_r = 0;
    float best_g = 0;
    float best_b = 0;
    float hit_x = 0;
    float hit_y = 0;
    float hit_z = 0;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    // TODO: TLAS=BLAS[]
    while (sp > 0) {
      const int node_idx = stack[--sp];
      if (node_idx == -1) continue;
      const Scene::BVH8Node& node = _scene->bvh8_nodes[node_idx];
      // Leaf node
      if (node.IsLeaf()) {
        // Convert prim_end to bitmask for register lanes
        float v0x[8], v0y[8], v0z[8], v1x[8], v1y[8], v1z[8], v2x[8], v2y[8],
            v2z[8];

        const int prim_end = node.tri_count;
        for (int p = 0; p < prim_end; ++p) {
          const int tri_idx = node.child_indices[p];
          // Fix this, I need the 0 to be representitive of the actual model
          // index, maybe can be encoded in the root node somehow?
          // See the other problem is that bvh8_nodes is data per model. So what
          // needs to happen is a Model/Mesh class needs to exist. If it is a
          // Mesh class maybe it could be shared to models, this would probably
          // require matrix math that to be honest, I don't think I care for
          // this project. I want do want to simply support multiple models

          auto& v0 = _scene->meshes[0]
                         .vertices[_scene->meshes[0].indices[tri_idx * 3 + 0]]
                         .pos;

          auto& v1 = _scene->meshes[0]
                         .vertices[_scene->meshes[0].indices[tri_idx * 3 + 1]]
                         .pos;

          auto& v2 = _scene->meshes[0]
                         .vertices[_scene->meshes[0].indices[tri_idx * 3 + 2]]
                         .pos;

          v0x[p] = v0.x;
          v0y[p] = v0.y;
          v0z[p] = v0.z;

          v1x[p] = v1.x;
          v1y[p] = v1.y;
          v1z[p] = v1.z;

          v2x[p] = v2.x;
          v2y[p] = v2.y;
          v2z[p] = v2.z;
        }

        const __m256 v0x_v = _mm256_load_ps(v0x);
        const __m256 v0y_v = _mm256_load_ps(v0y);
        const __m256 v0z_v = _mm256_load_ps(v0z);
        const __m256 v1x_v = _mm256_load_ps(v1x);
        const __m256 v1y_v = _mm256_load_ps(v1y);
        const __m256 v1z_v = _mm256_load_ps(v1z);
        const __m256 v2x_v = _mm256_load_ps(v2x);
        const __m256 v2y_v = _mm256_load_ps(v2y);
        const __m256 v2z_v = _mm256_load_ps(v2z);

        __m256 out_t, out_u, out_v;
        __m256 valid_v = simd::IntersectTriangle(
            ox, oy, oz, dx, dy, dz, v0x_v, v0y_v, v0z_v, v1x_v, v1y_v, v1z_v,
            v2x_v, v2y_v, v2z_v, t_min, t_best, out_t, out_u, out_v, false);

        alignas(32) float t_dists[8], t_u[8], t_v[8];
        _mm256_store_ps(t_dists, out_t);
        _mm256_store_ps(t_u, out_u);
        _mm256_store_ps(t_v, out_v);

        int valid = _mm256_movemask_ps(valid_v);

        int min_idx = 0;
        for (int i = 1; i < 8; ++i)
          if (valid & (1 << i) && t_dists[i] < t_dists[min_idx]) min_idx = i;

        const float val = t_dists[min_idx];
        if (val >= t_best) {
          continue;
        }

        t_best = val;

        const float mat_r = 0.0f;
        const float mat_g = 0.1f;
        const float mat_b = 1.0f;

        best_r = mat_r;
        best_g = mat_g;
        best_b = mat_b;

        // Barycentric interpolation
        float u = t_u[min_idx];
        float v = t_v[min_idx];

        const float w = 1 - u - v;
        hit_x = u * v0x[min_idx] + v * v1x[min_idx] + w * v2x[min_idx];
        hit_y = u * v0y[min_idx] + v * v1y[min_idx] + w * v2y[min_idx];
        hit_z = u * v0z[min_idx] + v * v1z[min_idx] + w * v2z[min_idx];

        continue;
      }

      __m256 child_hit_mask, child_t_near;
      simd::IntersectAABB8(ox, oy, oz, inv_dx, inv_dy, inv_dz, node.child_min_x,
                           node.child_min_y, node.child_min_z, node.child_max_x,
                           node.child_max_y, node.child_max_z, t_min, t_best,
                           child_hit_mask, child_t_near);

      struct HitChild {
        int idx;
        float tnear;
      };
      HitChild hit_children[8];
      int hit_count = 0;

      alignas(32) float t_near_arr[8];
      _mm256_store_ps(t_near_arr, child_t_near);
      int hits = _mm256_movemask_ps(child_hit_mask);

      for (int c = 0; c < 8; ++c) {
        if (hits & (1 << c)) {
          hit_children[hit_count++] = {node.child_indices[c], t_near_arr[c]};
        }
      }

      // Branchless distance sort
      if (hit_count > 1) {
        for (int i = 0; i < hit_count - 1; ++i) {
          for (int j = i + 1; j < hit_count; ++j) {
            const bool swap = hit_children[i].tnear > hit_children[j].tnear;
            const HitChild tmp = hit_children[i];
            hit_children[i] = swap ? hit_children[j] : hit_children[i];
            hit_children[j] = swap ? tmp : hit_children[j];
          }
        }
      }

      // Reverse queued for closed to be processed first
      for (int h = hit_count - 1; h >= 0; --h) {
        stack[sp++] = {hit_children[h].idx};
      }
    }

    bool missed = t_best >= rays->t_max[base];

    if (missed) {
      best_r = 0.7f + 0.3f * (0.5 + rays->direction_y[base] * 0.5f);
      best_g = best_r;
      best_b = 1.0f;
    }

    rays->color_r[base] = best_r;
    rays->color_g[base] = best_g;
    rays->color_b[base] = best_b;
  }
}

void Renderer::IntersectScene_Mt(Rays* rays) {
  struct RayBatch {
    uint32_t index;
    uint32_t count;
    uint8_t depth;
    uint8_t type;
  };

  struct RayBatchProcessor {
    // Can store any context you need
    Scene* scene;
    Rays* rays;

    void Process(const RayBatch& batch, WorkerContext& ctx) {
      for (uint32_t batch_id = 0; batch_id < batch.count; ++batch_id) {
        const int base = batch.index + batch_id;
        __m256 ox = _mm256_set1_ps(rays->origin_x[base]);
        __m256 oy = _mm256_set1_ps(rays->origin_y[base]);
        __m256 oz = _mm256_set1_ps(rays->origin_z[base]);
        __m256 dx = _mm256_set1_ps(rays->direction_x[base]);
        __m256 dy = _mm256_set1_ps(rays->direction_y[base]);
        __m256 dz = _mm256_set1_ps(rays->direction_z[base]);
        __m256 t_min = _mm256_set1_ps(rays->t_min[base]);
        __m256 t_max = _mm256_set1_ps(rays->t_max[base]);

        __m256 inv_dx = _mm256_rcp_ps(dx);
        __m256 inv_dy = _mm256_rcp_ps(dy);
        __m256 inv_dz = _mm256_rcp_ps(dz);

        float t_best = rays->t_max[base];
        float best_r = 0;
        float best_g = 0;
        float best_b = 0;
        float hit_x = 0;
        float hit_y = 0;
        float hit_z = 0;

        int stack[64];
        int sp = 0;
        stack[sp++] = 0;

        // TODO: TLAS=BLAS[]
        while (sp > 0) {
          const int node_idx = stack[--sp];
          if (node_idx == -1) continue;
          const Scene::BVH8Node& node = scene->bvh8_nodes[node_idx];
          // Leaf node
          if (node.IsLeaf()) {
            // Convert prim_end to bitmask for register lanes
            float v0x[8], v0y[8], v0z[8], v1x[8], v1y[8], v1z[8], v2x[8],
                v2y[8], v2z[8];

            const int prim_end = node.tri_count;
            for (int p = 0; p < prim_end; ++p) {
              const int tri_idx = node.child_indices[p];
              // Fix this, I need the 0 to be representitive of the actual model
              // index, maybe can be encoded in the root node somehow?
              // See the other problem is that bvh8_nodes is data per model. So
              // what needs to happen is a Model/Mesh class needs to exist. If
              // it is a Mesh class maybe it could be shared to models, this
              // would probably require matrix math that to be honest, I don't
              // think I care for this project. I want do want to simply support
              // multiple models

              auto& v0 =
                  scene->meshes[0]
                      .vertices[scene->meshes[0].indices[tri_idx * 3 + 0]]
                      .pos;

              auto& v1 =
                  scene->meshes[0]
                      .vertices[scene->meshes[0].indices[tri_idx * 3 + 1]]
                      .pos;

              auto& v2 =
                  scene->meshes[0]
                      .vertices[scene->meshes[0].indices[tri_idx * 3 + 2]]
                      .pos;

              v0x[p] = v0.x;
              v0y[p] = v0.y;
              v0z[p] = v0.z;

              v1x[p] = v1.x;
              v1y[p] = v1.y;
              v1z[p] = v1.z;

              v2x[p] = v2.x;
              v2y[p] = v2.y;
              v2z[p] = v2.z;
            }

            const __m256 v0x_v = _mm256_load_ps(v0x);
            const __m256 v0y_v = _mm256_load_ps(v0y);
            const __m256 v0z_v = _mm256_load_ps(v0z);
            const __m256 v1x_v = _mm256_load_ps(v1x);
            const __m256 v1y_v = _mm256_load_ps(v1y);
            const __m256 v1z_v = _mm256_load_ps(v1z);
            const __m256 v2x_v = _mm256_load_ps(v2x);
            const __m256 v2y_v = _mm256_load_ps(v2y);
            const __m256 v2z_v = _mm256_load_ps(v2z);

            __m256 out_t, out_u, out_v;
            __m256 valid_v = simd::IntersectTriangle(
                ox, oy, oz, dx, dy, dz, v0x_v, v0y_v, v0z_v, v1x_v, v1y_v,
                v1z_v, v2x_v, v2y_v, v2z_v, t_min, t_best, out_t, out_u, out_v,
                false);

            alignas(32) float t_dists[8], t_u[8], t_v[8];
            _mm256_store_ps(t_dists, out_t);
            _mm256_store_ps(t_u, out_u);
            _mm256_store_ps(t_v, out_v);

            int valid = _mm256_movemask_ps(valid_v);

            int min_idx = 0;
            for (int i = 1; i < 8; ++i)
              if (valid & (1 << i) && t_dists[i] < t_dists[min_idx])
                min_idx = i;

            const float val = t_dists[min_idx];
            if (val >= t_best) {
              continue;
            }

            t_best = val;

            const float mat_r = 0.0f;
            const float mat_g = 0.1f;
            const float mat_b = 1.0f;

            best_r = mat_r;
            best_g = mat_g;
            best_b = mat_b;

            // Barycentric interpolation
            float u = t_u[min_idx];
            float v = t_v[min_idx];

            const float w = 1 - u - v;
            hit_x = u * v0x[min_idx] + v * v1x[min_idx] + w * v2x[min_idx];
            hit_y = u * v0y[min_idx] + v * v1y[min_idx] + w * v2y[min_idx];
            hit_z = u * v0z[min_idx] + v * v1z[min_idx] + w * v2z[min_idx];

            continue;
          }

          __m256 child_hit_mask, child_t_near;
          simd::IntersectAABB8(ox, oy, oz, inv_dx, inv_dy, inv_dz,
                               node.child_min_x, node.child_min_y,
                               node.child_min_z, node.child_max_x,
                               node.child_max_y, node.child_max_z, t_min,
                               t_best, child_hit_mask, child_t_near);

          struct HitChild {
            int idx;
            float tnear;
          };
          HitChild hit_children[8];
          int hit_count = 0;

          alignas(32) float t_near_arr[8];
          _mm256_store_ps(t_near_arr, child_t_near);
          int hits = _mm256_movemask_ps(child_hit_mask);

          for (int c = 0; c < 8; ++c) {
            if (hits & (1 << c)) {
              hit_children[hit_count++] = {node.child_indices[c],
                                           t_near_arr[c]};
            }
          }

          // Branchless distance sort
          if (hit_count > 1) {
            for (int i = 0; i < hit_count - 1; ++i) {
              for (int j = i + 1; j < hit_count; ++j) {
                const bool swap = hit_children[i].tnear > hit_children[j].tnear;
                const HitChild tmp = hit_children[i];
                hit_children[i] = swap ? hit_children[j] : hit_children[i];
                hit_children[j] = swap ? tmp : hit_children[j];
              }
            }
          }

          // Reverse queued for closed to be processed first
          for (int h = hit_count - 1; h >= 0; --h) {
            stack[sp++] = {hit_children[h].idx};
          }
        }

        bool missed = t_best >= rays->t_max[base];

        if (missed) {
          best_r = 0.7f + 0.3f * (0.5 + rays->direction_y[base] * 0.5f);
          best_g = best_r;
          best_b = 1.0f;
        }

        rays->color_r[base] = best_r;
        rays->color_g[base] = best_g;
        rays->color_b[base] = best_b;

        // ctx.rays_traced.fetch_add(1, std::memory_order_relaxed);
        // ctx.triangles_tested.fetch_add(8, std::memory_order_relaxed);
      }
    }
  };

  ThreadPool<RayBatch, RayBatchProcessor> thread_pool(
      RayBatchProcessor(_scene, rays));
  thread_pool.Start();

  uint32_t num_workers = thread_pool.NumWorkers();
  const uint32_t BATCH_SIZE = 64;
  uint32_t total_rays = rays->count;
  for (uint32_t i = 0; i < total_rays; i += BATCH_SIZE) {
    RayBatch batch{
        i,                                     // start_index
        std::min(BATCH_SIZE, total_rays - i),  // count
        0                                      // depth
    };

    // Round-robin to workers
    uint32_t worker = (i / BATCH_SIZE) % num_workers;
    thread_pool.SubmitBatch(worker, batch);
  }

  thread_pool.WaitForCompletion();
}

void Renderer::CompactRays(Rays* rays, float* image, float contrib_factor) {
  int writeIdx = 0;
  for (int i = 0; i < rays->count; i++) {
    int pixel = rays->pixel_index[i] * CHANNEL_NUM;
    image[pixel + 0] += rays->color_r[i] * contrib_factor;
    image[pixel + 1] += rays->color_g[i] * contrib_factor;
    image[pixel + 2] += rays->color_b[i] * contrib_factor;
  }
  rays->count = writeIdx;
}