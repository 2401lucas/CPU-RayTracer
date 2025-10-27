#pragma once
#include <immintrin.h>

namespace simd {
inline int ActiveMaskCount(__m256 mask) {
  return __popcnt(_mm256_movemask_ps(mask));
}

inline __m256 Clamp(__m256 v, __m256 min_v, __m256 max_v) {
  return _mm256_min_ps(_mm256_max_ps(v, min_v), max_v);
}

inline __m256 Dot(__m256 ax, __m256 ay, __m256 az, __m256 bx, __m256 by,
                  __m256 bz) {
  return _mm256_fmadd_ps(az, bz,
                         _mm256_fmadd_ps(ay, by, _mm256_mul_ps(ax, bx)));
}

inline void Cross(__m256 ax, __m256 ay, __m256 az, __m256 bx, __m256 by,
                  __m256 bz, __m256& rx, __m256& ry, __m256& rz) {
  rx = _mm256_sub_ps(_mm256_mul_ps(ay, bz), _mm256_mul_ps(az, by));
  ry = _mm256_sub_ps(_mm256_mul_ps(az, bx), _mm256_mul_ps(ax, bz));
  rz = _mm256_sub_ps(_mm256_mul_ps(ax, by), _mm256_mul_ps(ay, bx));
}

inline void Normalize(__m256& x, __m256& y, __m256& z) {
  __m256 len_sq = Dot(x, y, z, x, y, z);
  __m256 inv_len = _mm256_rsqrt_ps(len_sq);
  x = _mm256_mul_ps(x, inv_len);
  y = _mm256_mul_ps(y, inv_len);
  z = _mm256_mul_ps(z, inv_len);
}

inline __m256 Random(__m256i& state) {
  // Xorshift algo
  state = _mm256_xor_si256(state, _mm256_slli_epi32(state, 13));
  state = _mm256_xor_si256(state, _mm256_srli_epi32(state, 17));
  state = _mm256_xor_si256(state, _mm256_slli_epi32(state, 5));

  // Convert to float [0, 1]
  __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
  __m256i t_masked = _mm256_and_si256(state, mask);
  return _mm256_mul_ps(_mm256_cvtepi32_ps(t_masked),
                       _mm256_set1_ps(1.0f / 2147483647.0f));
}

inline void RandomInUnitDisc(__m256i& state, __m256& out_x, __m256& out_y) {
  __m256 two = _mm256_set1_ps(2.0f);
  __m256 one = _mm256_set1_ps(1.0f);

  // Generate random points in [-1, 1] square and reject if outside unit circle
  __m256 r1 = _mm256_sub_ps(_mm256_mul_ps(Random(state), two), one);
  __m256 r2 = _mm256_sub_ps(_mm256_mul_ps(Random(state), two), one);

  __m256 len_sq = _mm256_add_ps(_mm256_mul_ps(r1, r1), _mm256_mul_ps(r2, r2));
  __m256 mask = _mm256_cmp_ps(len_sq, one, _CMP_LE_OQ);

  // Use rejection sampling results (hack - TODO: full rejection loop)
  out_x = _mm256_and_ps(r1, mask);
  out_y = _mm256_and_ps(r2, mask);
}

// Basic Sphere intersection : 8rays x 1 sphere
// returns hit mask
// out_t = intersection dist
inline __m256 IntersectSphere(const __m256 ox, const __m256 oy, const __m256 oz,
                              const __m256 dx, const __m256 dy, const __m256 dz,
                              const float sx, const float sy, const float sz,
                              const float sr, const __m256 t_min,
                              const __m256 t_max, __m256& out_t,
                              float epsilon = 1e-8f) {
  __m256 cx = _mm256_set1_ps(sx);
  __m256 cy = _mm256_set1_ps(sy);
  __m256 cz = _mm256_set1_ps(sz);
  __m256 r = _mm256_set1_ps(sr);

  // oc = origin - center
  __m256 ocx = _mm256_sub_ps(ox, cx);
  __m256 ocy = _mm256_sub_ps(oy, cy);
  __m256 ocz = _mm256_sub_ps(oz, cz);

  // quadratic coefficients
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

  // disc > 0
  __m256 discMask = _mm256_cmp_ps(disc, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
  // no lane has intersection with this sphere
  if (_mm256_movemask_ps(discMask) == 0) return _mm256_setzero_ps();

  // sqrt(disc)
  __m256 sqd = _mm256_sqrt_ps(disc);
  __m256 t0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), b),
                            sqd);  // -b - sqrt
  __m256 t1 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), b),
                            sqd);  // -b + sqrt

  // prefer near positive root
  // if t0 > t_min, t = t0, else t1
  __m256 t0_gt_t_min = _mm256_cmp_ps(t0, t_min, _CMP_GT_OQ);
  __m256 t = _mm256_blendv_ps(t1, t0, t0_gt_t_min);

  // valid = discMask && (t_candidate > t_min) && (t_candidate < t_best)
  __m256 gt_t_min = _mm256_cmp_ps(t, t_min, _CMP_GT_OQ);
  __m256 lt_t_best = _mm256_cmp_ps(t, t_max, _CMP_LT_OQ);
  __m256 valid = _mm256_and_ps(discMask, _mm256_and_ps(gt_t_min, lt_t_best));

  out_t = _mm256_blendv_ps(_mm256_set1_ps(1e30f), t, valid);

  return valid;
}

// Möller–Trumbore tri intersection algorithm 8rays x 1 tri
// returns hit mask
// out_t = intersection dist
// out_u, out_v = barycentric coords
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
inline __m256 IntersectTriangle(
    const __m256 ox, const __m256 oy, const __m256 oz, const __m256 dx,
    const __m256 dy, const __m256 dz, const float v0x, const float v0y,
    const float v0z, const float v1x, const float v1y, const float v1z,
    const float v2x, const float v2y, const float v2z, const __m256 tMin,
    const __m256 tMax, __m256& out_t, __m256& out_u, __m256& out_v,
    bool backface_cull = false, float epsilon = 1e-8f) {
  __m256 v0x_v = _mm256_set1_ps(v0x);
  __m256 v0y_v = _mm256_set1_ps(v0y);
  __m256 v0z_v = _mm256_set1_ps(v0z);

  __m256 v1x_v = _mm256_set1_ps(v1x);
  __m256 v1y_v = _mm256_set1_ps(v1y);
  __m256 v1z_v = _mm256_set1_ps(v1z);

  __m256 v2x_v = _mm256_set1_ps(v2x);
  __m256 v2y_v = _mm256_set1_ps(v2y);
  __m256 v2z_v = _mm256_set1_ps(v2z);

  // edges
  __m256 e1x = _mm256_sub_ps(v1x_v, v0x_v);
  __m256 e1y = _mm256_sub_ps(v1y_v, v0y_v);
  __m256 e1z = _mm256_sub_ps(v1z_v, v0z_v);

  __m256 e2x = _mm256_sub_ps(v2x_v, v0x_v);
  __m256 e2y = _mm256_sub_ps(v2y_v, v0y_v);
  __m256 e2z = _mm256_sub_ps(v2z_v, v0z_v);

  // p = cross(dir, e2)
  __m256 px = _mm256_sub_ps(_mm256_mul_ps(dy, e2z), _mm256_mul_ps(dz, e2y));
  __m256 py = _mm256_sub_ps(_mm256_mul_ps(dz, e2x), _mm256_mul_ps(dx, e2z));
  __m256 pz = _mm256_sub_ps(_mm256_mul_ps(dx, e2y), _mm256_mul_ps(dy, e2x));

  // det = dot(e1, p)
  __m256 det = _mm256_add_ps(
      _mm256_mul_ps(e1x, px),
      _mm256_add_ps(_mm256_mul_ps(e1y, py), _mm256_mul_ps(e1z, pz)));

  const __m256 eps_v = _mm256_set1_ps(epsilon);
  __m256 det_mask;
  if (backface_cull) {
    // det > epsilon
    det_mask = _mm256_cmp_ps(det, eps_v, _CMP_GT_OQ);
  } else {
    // abs(det) > epsilon
    __m256 abs_det =
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), det);  // abs (by clearing sign)
    det_mask = _mm256_cmp_ps(abs_det, eps_v, _CMP_GT_OQ);
  }

  // If no lane has det_mask, early out
  if (_mm256_movemask_ps(det_mask) == 0) {
    out_t = _mm256_set1_ps(1e30f);
    out_u = out_v = _mm256_setzero_ps();
    return _mm256_setzero_ps();
  }

  // invDet = 1.0 / det
  __m256 invDet = _mm256_div_ps(_mm256_set1_ps(1.0f), det);

  // t = origin - v0
  __m256 tx = _mm256_sub_ps(ox, v0x_v);
  __m256 ty = _mm256_sub_ps(oy, v0y_v);
  __m256 tz = _mm256_sub_ps(oz, v0z_v);

  // u = dot(t, p) * invDet
  __m256 u = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(tx, px),
                                         _mm256_add_ps(_mm256_mul_ps(ty, py),
                                                       _mm256_mul_ps(tz, pz))),
                           invDet);

  // u test: for backface_cull: u<0 || u>1 => reject
  __m256 zero_v = _mm256_setzero_ps();
  __m256 one_v = _mm256_set1_ps(1.0f);

  __m256 u_ge0 = _mm256_cmp_ps(u, zero_v, _CMP_GE_OQ);  // u >= 0
  __m256 u_le1 = _mm256_cmp_ps(u, one_v, _CMP_LE_OQ);   // u <= 1
  __m256 u_mask = _mm256_and_ps(u_ge0, u_le1);

  // q = cross(t, e1)
  __m256 qx = _mm256_sub_ps(_mm256_mul_ps(ty, e1z), _mm256_mul_ps(tz, e1y));
  __m256 qy = _mm256_sub_ps(_mm256_mul_ps(tz, e1x), _mm256_mul_ps(tx, e1z));
  __m256 qz = _mm256_sub_ps(_mm256_mul_ps(tx, e1y), _mm256_mul_ps(ty, e1x));

  // v = dot(dir, q) * invDet
  __m256 v = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(dx, qx),
                                         _mm256_add_ps(_mm256_mul_ps(dy, qy),
                                                       _mm256_mul_ps(dz, qz))),
                           invDet);

  // v test: v >= 0 && u+v <=1
  __m256 v_ge0 = _mm256_cmp_ps(v, zero_v, _CMP_GE_OQ);
  __m256 uv_sum = _mm256_add_ps(u, v);
  __m256 uv_le1 = _mm256_cmp_ps(uv_sum, one_v, _CMP_LE_OQ);

  __m256 uv_mask = _mm256_and_ps(v_ge0, uv_le1);

  // t = dot(e2, q) * invDet
  __m256 t = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(e2x, qx),
                                         _mm256_add_ps(_mm256_mul_ps(e2y, qy),
                                                       _mm256_mul_ps(e2z, qz))),
                           invDet);

  // t test: t > tMin && t < tMax
  __m256 gt_tmin = _mm256_cmp_ps(t, tMin, _CMP_GT_OQ);
  __m256 lt_tmax = _mm256_cmp_ps(t, tMax, _CMP_LT_OQ);
  __m256 t_mask = _mm256_and_ps(gt_tmin, lt_tmax);

  // mask = det_mask && u_mask && uv_mask && t_mask
  __m256 mask = _mm256_and_ps(
      det_mask, _mm256_and_ps(u_mask, _mm256_and_ps(uv_mask, t_mask)));

  out_t = _mm256_blendv_ps(_mm256_set1_ps(1e30f), t, mask);
  out_u = _mm256_blendv_ps(zero_v, u, mask);
  out_v = _mm256_blendv_ps(zero_v, v, mask);

  return mask;
}

// Möller–Trumbore tri intersection algorithm 1rays x 8 tri
// returns hit mask
// out_t = intersection dist
// out_u, out_v = barycentric coords
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
inline __m256 IntersectTriangle(
    const __m256 ox, const __m256 oy, const __m256 oz, const __m256 dx,
    const __m256 dy, const __m256 dz, const __m256 v0x, const __m256 v0y,
    const __m256 v0z, const __m256 v1x, const __m256 v1y, const __m256 v1z,
    const __m256 v2x, const __m256 v2y, const __m256 v2z, const __m256 t_min,
    const float tMax, __m256& out_t, __m256& out_u, __m256& out_v,
    bool backface_cull = true, float epsilon = 1e-8f) {
  __m256 t_max_v = _mm256_set1_ps(tMax);

  __m256 e1x = _mm256_sub_ps(v1x, v0x);
  __m256 e1y = _mm256_sub_ps(v1y, v0y);
  __m256 e1z = _mm256_sub_ps(v1z, v0z);

  __m256 e2x = _mm256_sub_ps(v2x, v0x);
  __m256 e2y = _mm256_sub_ps(v2y, v0y);
  __m256 e2z = _mm256_sub_ps(v2z, v0z);

  __m256 px, py, pz;
  Cross(dx, dy, dz, e2x, e2y, e2z, px, py, pz);

  __m256 det = Dot(e1x, e1y, e1z, px, py, pz);

  const __m256 eps_v = _mm256_set1_ps(epsilon);
  __m256 det_mask;
  if (backface_cull) {
    det_mask = _mm256_cmp_ps(det, eps_v, _CMP_GT_OQ);
  } else {
    __m256 abs_det =
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), det);  // clearing sign for abs
    det_mask = _mm256_cmp_ps(abs_det, eps_v, _CMP_GT_OQ);
  }

  if (_mm256_movemask_ps(det_mask) == 0) {
    out_t = _mm256_set1_ps(FLT_MAX);
    out_u = out_v = _mm256_setzero_ps();
    return _mm256_setzero_ps();
  }

  __m256 invDet = _mm256_div_ps(_mm256_set1_ps(1.0f), det);

  __m256 tx = _mm256_sub_ps(ox, v0x);
  __m256 ty = _mm256_sub_ps(oy, v0y);
  __m256 tz = _mm256_sub_ps(oz, v0z);

  __m256 u = _mm256_mul_ps(Dot(tx, ty, tz, px, py, pz), invDet);

  __m256 zero_v = _mm256_setzero_ps();
  __m256 one_v = _mm256_set1_ps(1.0f);

  __m256 u_ge0 = _mm256_cmp_ps(u, zero_v, _CMP_GE_OQ);
  __m256 u_le1 = _mm256_cmp_ps(u, one_v, _CMP_LE_OQ);
  __m256 u_mask = _mm256_and_ps(u_ge0, u_le1);

  __m256 qx, qy, qz;
  Cross(tx, ty, tz, e1x, e1y, e1z, qx, qy, qz);

  __m256 v = _mm256_mul_ps(Dot(dx, dy, dz, qx, qy, qz), invDet);

  __m256 v_ge0 = _mm256_cmp_ps(v, zero_v, _CMP_GE_OQ);
  __m256 uv_sum = _mm256_add_ps(u, v);
  __m256 uv_le1 = _mm256_cmp_ps(uv_sum, one_v, _CMP_LE_OQ);

  __m256 uv_mask = _mm256_and_ps(v_ge0, uv_le1);

  __m256 t = _mm256_mul_ps(Dot(e2x, e2y, e2z, qx, qy, qz), invDet);

  __m256 gt_tmin = _mm256_cmp_ps(t, t_min, _CMP_GT_OQ);
  __m256 lt_tmax = _mm256_cmp_ps(t, t_max_v, _CMP_LT_OQ);
  __m256 t_mask = _mm256_and_ps(gt_tmin, lt_tmax);

  __m256 mask = _mm256_and_ps(
      det_mask, _mm256_and_ps(u_mask, _mm256_and_ps(uv_mask, t_mask)));

  out_t = _mm256_blendv_ps(_mm256_set1_ps(FLT_MAX), t, mask);
  out_u = _mm256_blendv_ps(zero_v, u, mask);
  out_v = _mm256_blendv_ps(zero_v, v, mask);

  return mask;
}

static inline void IntersectAABB8(
    const __m256 ox, const __m256 oy, const __m256 oz, const __m256 invDx,
    const __m256 invDy, const __m256 invDz, const float const* child_min_x,
    const float const* child_min_y, const float const* child_min_z,
    const float const* child_max_x, const float const* child_max_y,
    const float const* child_max_z, const __m256 t_min, const float t_max,
    __m256& child_mask_out, __m256& child_t_near_out) {
  __m256 t_max_v = _mm256_set1_ps(t_max);
  __m256 min_x = _mm256_load_ps(child_min_x);
  __m256 max_x = _mm256_load_ps(child_max_x);
  __m256 min_y = _mm256_load_ps(child_min_y);
  __m256 max_y = _mm256_load_ps(child_max_y);
  __m256 min_z = _mm256_load_ps(child_min_z);
  __m256 max_z = _mm256_load_ps(child_max_z);

  __m256 tmin_x = _mm256_mul_ps(_mm256_sub_ps(min_x, ox), invDx);
  __m256 tmax_x = _mm256_mul_ps(_mm256_sub_ps(max_x, ox), invDx);
  __m256 tx_near = _mm256_min_ps(tmin_x, tmax_x);
  __m256 tx_far = _mm256_max_ps(tmin_x, tmax_x);

  __m256 tmin_y = _mm256_mul_ps(_mm256_sub_ps(min_y, oy), invDy);
  __m256 tmax_y = _mm256_mul_ps(_mm256_sub_ps(max_y, oy), invDy);
  __m256 ty_near = _mm256_min_ps(tmin_y, tmax_y);
  __m256 ty_far = _mm256_max_ps(tmin_y, tmax_y);

  __m256 tmin_z = _mm256_mul_ps(_mm256_sub_ps(min_z, oz), invDz);
  __m256 tmax_z = _mm256_mul_ps(_mm256_sub_ps(max_z, oz), invDz);
  __m256 tz_near = _mm256_min_ps(tmin_z, tmax_z);
  __m256 tz_far = _mm256_max_ps(tmin_z, tmax_z);

  __m256 t_near = _mm256_max_ps(tx_near, _mm256_max_ps(ty_near, tz_near));
  __m256 t_far = _mm256_min_ps(tx_far, _mm256_min_ps(ty_far, tz_far));

  child_mask_out =
      _mm256_and_ps(_mm256_cmp_ps(t_near, t_far, _CMP_LE_OQ),
                    _mm256_and_ps(_mm256_cmp_ps(t_far, t_min, _CMP_GE_OQ),
                                  _mm256_cmp_ps(t_near, t_max_v, _CMP_LE_OQ)));

  child_t_near_out = t_near;
}
}  // namespace simd