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
  __m256 x = _mm256_mul_ps(ax, bx);
  __m256 y = _mm256_mul_ps(ay, by);
  __m256 z = _mm256_mul_ps(az, bz);
  return _mm256_add_ps(x, _mm256_add_ps(y, z));
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

// Basic Sphere intersection : 8rays x 1 sphere
// returns hit mask
// out_t = intersection dist
inline __m256 IntersectSphere_NoBVH(const __m256 ox, const __m256 oy,
                                    const __m256 oz, const __m256 dx,
                                    const __m256 dy, const __m256 dz,
                                    const float sx, const float sy,
                                    const float sz, const float sr,
                                    const __m256 t_min, const __m256 t_max,
                                    __m256& out_t, float epsilon = 1e-8f) {
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
inline __m256 IntersectTriangle_NoBVH(
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
}  // namespace simd