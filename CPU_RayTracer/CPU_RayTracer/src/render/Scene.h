#pragma once
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <glm/glm.hpp>

class Scene {
 public:
  // OLD FOR NON BVH RAYTRACING--------------------------------------------
  struct Spheres {
    int capacity;
    int count;
    float* center_x;
    float* center_y;
    float* center_z;
    float* radius;
    float* mat_r;
    float* mat_g;
    float* mat_b;
  } spheres;

  // Currently vbuffs are ordered so that v0 + v1 +v2 makes a face. This is not
  // the most memory efficient but it lets us omit an index buffer and makes
  // processing with simd easier.
  struct Tris {
    int capacity;
    int count;
    float* v0_x;
    float* v0_y;
    float* v0_z;
    float* v1_x;
    float* v1_y;
    float* v1_z;
    float* v2_x;
    float* v2_y;
    float* v2_z;
    float* centroid_x;
    float* centroid_y;
    float* centroid_z;
  } tris;
  // ----------------------------------------------------------------------

  struct Vertex {
    glm::vec3 pos;
    Vertex(float x, float y, float z) { pos = glm::vec3(x, y, z); }
  };

  struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

  };

  struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() : min(glm::vec3(0.0f)), max(glm::vec3(0.0f)) {}

    AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    void Expand(const AABB& other) {
      min = glm::min(min, other.min);
      max = glm::max(max, other.max);
    }

    void Expand(const glm::vec3& point) {
      min = glm::min(min, point);
      max = glm::max(max, point);
    }

    glm::vec3 Extent() const { return max - min; }

    float SurfaceArea() const {
      glm::vec3 extent = max - min;
      if (extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) {
        return 0.0f;  // Invalid box
      }
      return 2.0f *
             (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }

    bool IsValid() const {
      return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }

    static AABB Invalid() {
      return AABB(glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX),
                  glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    }
  };

  struct BVHNode {
    AABB bounds;
    int first;
    int tri_count;  // 0 = internal
  };

  struct alignas(32) BVH8Node {
    float child_min_x[8];
    float child_min_y[8];
    float child_min_z[8];
    float child_max_x[8];
    float child_max_y[8];
    float child_max_z[8];

    // Child node indices or primitive indices
    int child_indices[8];

    int tri_count;
    int padding[3];

    // Helper to check if this is a leaf
    bool IsLeaf() const { return tri_count > 0; }

    // Create invalid node
    static BVH8Node Invalid() {
      BVH8Node node;
      node.tri_count = 0;
      for (int i = 0; i < 8; i++) {
        node.child_min_x[i] = FLT_MAX;
        node.child_min_y[i] = FLT_MAX;
        node.child_min_z[i] = FLT_MAX;
        node.child_max_x[i] = -FLT_MAX;
        node.child_max_y[i] = -FLT_MAX;
        node.child_max_z[i] = -FLT_MAX;
        node.child_indices[i] = -1;
      }
      return node;
    }
  };

  std::vector<Mesh> meshes;
  std::vector<BVHNode> bvh_nodes;
  std::vector<BVH8Node> bvh8_nodes;

 public:
  Scene(int max_spheres, int max_faces);
  ~Scene();

  void LoadScene(const char* filepath);
  void LoadModel(const char* filepath, aiVector3f offset = {0, 0, 0});

 private:
  void BuildBLAS();
  int BuildBVH(std::vector<int>& tri_indices,
               const std::vector<AABB>& tri_bounds,
               const std::vector<glm::vec3>& tri_centroids, int start, int end);
  int BuildBVH8(std::vector<int>& tri_indices,
                const std::vector<AABB>& tri_bounds,
                const std::vector<glm::vec3>& tri_centroids, int start,
                int end);
};