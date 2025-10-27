#include "Scene.h"

#include <iostream>
#include <numeric>
#include <queue>

Scene::Scene(int max_spheres, int max_tris) {
  spheres.capacity = max_spheres;
  spheres.count = 0;
  spheres.center_x = (float*)_mm_malloc(max_spheres * sizeof(float), 32);
  spheres.center_y = (float*)_mm_malloc(max_spheres * sizeof(float), 32);
  spheres.center_z = (float*)_mm_malloc(max_spheres * sizeof(float), 32);
  spheres.radius = (float*)_mm_malloc(max_spheres * sizeof(float), 32);
  spheres.mat_r = (float*)_mm_malloc(max_spheres * sizeof(float), 32);
  spheres.mat_g = (float*)_mm_malloc(max_spheres * sizeof(float), 32);
  spheres.mat_b = (float*)_mm_malloc(max_spheres * sizeof(float), 32);

  tris.capacity = max_tris;
  tris.count = 0;
  tris.v0_x = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v0_y = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v0_z = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v1_x = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v1_y = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v1_z = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v2_x = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v2_y = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.v2_z = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.centroid_x = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.centroid_y = (float*)_mm_malloc(max_tris * sizeof(float), 32);
  tris.centroid_z = (float*)_mm_malloc(max_tris * sizeof(float), 32);
}
Scene::~Scene() {
  _mm_free(spheres.center_x);
  _mm_free(spheres.center_y);
  _mm_free(spheres.center_z);
  _mm_free(spheres.radius);
  _mm_free(spheres.mat_r);
  _mm_free(spheres.mat_g);
  _mm_free(spheres.mat_b);

  _mm_free(tris.v0_x);
  _mm_free(tris.v0_y);
  _mm_free(tris.v0_z);
  _mm_free(tris.v1_x);
  _mm_free(tris.v1_y);
  _mm_free(tris.v1_z);
  _mm_free(tris.v2_x);
  _mm_free(tris.v2_y);
  _mm_free(tris.v2_z);
  _mm_free(tris.centroid_x);
  _mm_free(tris.centroid_y);
  _mm_free(tris.centroid_z);
}

void Scene::LoadScene(const char* filepath) {
  spheres.count = 0;

  spheres.center_x[0] = 0.f;
  spheres.center_y[0] = 0.f;
  spheres.center_z[0] = 1.f;
  spheres.radius[0] = .1f;
  spheres.mat_r[0] = 1.f;
  spheres.mat_g[0] = 0.f;
  spheres.mat_b[0] = 0.f;

  spheres.center_x[1] = 0.f;
  spheres.center_y[1] = 0.f;
  spheres.center_z[1] = 8.f;
  spheres.radius[1] = 3.1f;
  spheres.mat_r[1] = 1.f;
  spheres.mat_g[1] = 0.f;
  spheres.mat_b[1] = 0.f;

  LoadModel("assets/Cube/Cube.gltf", {0, 0, 0});

  BuildBLAS();
}

// Loads a model at filepath into the primary vertex buffers
void Scene::LoadModel(const char* filepath, aiVector3f offset) {
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(
      filepath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
                    aiProcess_MakeLeftHanded);
  if (scene == nullptr) {
    std::cerr << importer.GetErrorString() << std::endl;
    return;
  }

#ifdef DEBUG_MODEL
  std::cout << "Filepath Name: " << filepath << std::endl;
  std::cout << "Model Name: " << scene->mName.C_Str() << std::endl;
  std::cout << "Mesh Count: " << scene->mNumMeshes << std::endl;
  std::cout << "Texture Count: " << scene->mNumTextures << std::endl;
  std::cout << "Materials Count: " << scene->mNumMaterials << std::endl;
#endif
  // OLD FOR NON BVH RAYTRACING--------------------------------------------
  for (uint32_t m = 0; m < scene->mNumMeshes; m++) {
    auto& mesh = scene->mMeshes[m];
    if (mesh->HasFaces() && mesh->HasPositions()) {
      for (size_t i = 0; i < mesh->mNumFaces; i++) {
        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
          auto vert0 = (mesh->mVertices[mesh->mFaces[j].mIndices[0]]);
          auto vert1 = (mesh->mVertices[mesh->mFaces[j].mIndices[1]]);
          auto vert2 = (mesh->mVertices[mesh->mFaces[j].mIndices[2]]);
          tris.v0_x[tris.count] = vert0.x + offset.x;
          tris.v0_y[tris.count] = vert0.y + offset.y;
          tris.v0_z[tris.count] = vert0.z + offset.z;
          tris.v1_x[tris.count] = vert1.x + offset.x;
          tris.v1_y[tris.count] = vert1.y + offset.y;
          tris.v1_z[tris.count] = vert1.z + offset.z;
          tris.v2_x[tris.count] = vert2.x + offset.x;
          tris.v2_y[tris.count] = vert2.y + offset.y;
          tris.v2_z[tris.count] = vert2.z + offset.z;

          tris.count++;
        }
      }
    }
  }
  // ----------------------------------------------------------------------

  for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
    Mesh new_mesh;
    auto& mesh = scene->mMeshes[i];
    if (mesh->HasPositions()) {
      for (size_t j = 0; j < mesh->mNumVertices; j++) {
        new_mesh.vertices.push_back({mesh->mVertices[j].x + offset.x,
                                     mesh->mVertices[j].y + offset.y,
                                     mesh->mVertices[j].z + offset.z});
      }
    }
    if (mesh->HasFaces()) {
      for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
        new_mesh.indices.push_back(mesh->mFaces[j].mIndices[0]);
        new_mesh.indices.push_back(mesh->mFaces[j].mIndices[1]);
        new_mesh.indices.push_back(mesh->mFaces[j].mIndices[2]);
      }
    }

    meshes.push_back(new_mesh);
  }
}
// How could the TLAS interact with BLAS
// If Each BLAS is stored in a unique container, this means we don't need to
// hold the parent node as it would always be 0.
// Each TLAS would then hold the index/reference to the BLAS & the bounds of the
// Node How would each TLAS be subdivided? Assuming 2 BLAS of hugely different
// sizes, maybe each BLAS should be subdiveded into multiple BLAS, this could be
// regulated by the bounds min/max.
void Scene::BuildBLAS() {
  bvh8_nodes.clear();
  // Builds BLAS
  for (size_t i = 0; i < meshes.size(); ++i) {
    const Mesh& mesh = meshes[i];
    std::vector<int> tri_indices(mesh.indices.size() / 3);
    std::iota(tri_indices.begin(), tri_indices.end(), 0);

    std::vector<AABB> tri_bounds;
    std::vector<glm::vec3> tri_centroids;
    tri_bounds.reserve(tri_indices.size());
    tri_centroids.reserve(tri_indices.size());

    for (size_t j = 0; j < tri_indices.size(); ++j) {
      uint32_t i0 = mesh.indices[j * 3 + 0];
      uint32_t i1 = mesh.indices[j * 3 + 1];
      uint32_t i2 = mesh.indices[j * 3 + 2];
      glm::vec3 v0 = mesh.vertices[i0].pos;
      glm::vec3 v1 = mesh.vertices[i1].pos;
      glm::vec3 v2 = mesh.vertices[i2].pos;

      AABB b{glm::min(glm::min(v0, v1), v2), glm::max(glm::max(v0, v1), v2)};
      tri_bounds.push_back(b);
      tri_centroids.push_back((v0 + v1 + v2) * 0.3333f);
    }

    BuildBVH8(tri_indices, tri_bounds, tri_centroids, 0, tri_indices.size());
    bvh8_nodes.shrink_to_fit();
  }
}

int Scene::BuildBVH(std::vector<int>& tri_indices,
                    const std::vector<AABB>& tri_bounds,
                    const std::vector<glm::vec3>& tri_centroids, int start,
                    int end) {
  BVHNode node;
  AABB bounds = tri_bounds[tri_indices[start]];

  for (int i = start + 1; i < end; i++)
    bounds.Expand(tri_bounds[tri_indices[i]]);
  node.bounds = bounds;

  int node_index = bvh_nodes.size();
  bvh_nodes.push_back(node);

  int tri_count = end - start;
  if (tri_count <= 8) {  // leaf threshold
    node.first = start;
    node.tri_count = tri_count;
    bvh_nodes[node_index] = node;
    return node_index;
  }

  // Split
  glm::vec3 extent = bounds.Extent();
  int axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2)
                                 : (extent.y > extent.z ? 1 : 2);

  float mid = 0.0f;
  for (int i = 0; i < end; i++) {
    mid += tri_centroids[tri_indices[i]][axis];
  }

  int mid_index =
      std::partition(tri_indices.begin() + start, tri_indices.begin() + end,
                     [&](int i) { return tri_centroids[i][axis] < mid; }) -
      tri_indices.begin();

  if (mid_index == start || mid_index == end) {
    mid_index = start + tri_count / 2;
  }

  node.first = (int)bvh_nodes.size();
  node.tri_count = 0;
  bvh_nodes[node_index] = node;

  BuildBVH(tri_indices, tri_bounds, tri_centroids, start, mid_index);
  BuildBVH(tri_indices, tri_bounds, tri_centroids, mid_index, end);

  return node_index;
}

int Scene::BuildBVH8(std::vector<int>& tri_indices,
                     const std::vector<AABB>& tri_bounds,
                     const std::vector<glm::vec3>& tri_centroids, int start,
                     int end) {
  struct BuildTask {
    int node_index;
    int start;
    int end;
  };

  struct Bin {
    AABB bounds;
    int count = 0;

    void Add(const AABB& tri_bounds) {
      if (count == 0) {
        bounds = tri_bounds;
      } else {
        bounds.Expand(tri_bounds);
      }
      count++;
    }
  };

  const int NUM_BINS = 16;
  const float TRAVERSAL_COST = 1.0f;
  const float INTERSECTION_COST = 0.2f;

  std::queue<BuildTask> task_queue;

  bvh8_nodes.reserve(tri_indices.size() * 2);

  // Create root node
  AABB root_bounds = tri_bounds[tri_indices[start]];
  AABB centroid_bounds;
  centroid_bounds.min = centroid_bounds.max = tri_centroids[tri_indices[start]];

  for (int i = start + 1; i < end; i++) {
    root_bounds.Expand(tri_bounds[tri_indices[i]]);
    centroid_bounds.Expand(tri_centroids[tri_indices[i]]);
  }

  int root_index = bvh8_nodes.size();
  bvh8_nodes.push_back(BVH8Node::Invalid());
  task_queue.push({root_index, start, end});

  // Process all nodes iteratively
  while (!task_queue.empty()) {
    BuildTask task = task_queue.front();
    task_queue.pop();

    int tri_count = task.end - task.start;

    AABB node_bounds = tri_bounds[tri_indices[task.start]];
    AABB centroid_bounds;
    centroid_bounds.min = centroid_bounds.max =
        tri_centroids[tri_indices[task.start]];
    for (int i = task.start + 1; i < task.end; i++) {
      centroid_bounds.Expand(tri_centroids[tri_indices[i]]);
    }

    // Check if this should be a leaf
    if (tri_count <= 8) {
      bvh8_nodes[task.node_index].tri_count = tri_count;
      for (int i = 0; i < tri_count; i++)
        bvh8_nodes[task.node_index].child_indices[i] = task.start + i;
      for (int i = tri_count; i < 8; i++)
        bvh8_nodes[task.node_index].child_indices[i] = -1;
      continue;
    }

    glm::vec3 centroid_extent = centroid_bounds.Extent();
    if (centroid_extent.x < 1e-6f && centroid_extent.y < 1e-6f &&
        centroid_extent.z < 1e-6f) {
      bvh8_nodes[task.node_index].tri_count = tri_count;
      for (int i = 0; i < tri_count; i++) {
        bvh8_nodes[task.node_index].child_indices[i] = task.start + i;
      }
      for (int i = tri_count; i < 8; i++) {
        bvh8_nodes[task.node_index].child_indices[i] = -1;
      }
      continue;
    }

    // Find best split using SAH
    float best_cost = FLT_MAX;
    int best_axis = 0;
    int best_num_children = 2;
    int best_split_indices[8];

    // Try each axis
    for (int axis = 0; axis < 3; axis++) {
      float centroid_min = centroid_bounds.min[axis];
      float centroid_max = centroid_bounds.max[axis];

      if (centroid_max - centroid_min < 1e-6f) continue;

      for (int num_children = 2; num_children <= 8; num_children++) {
        Bin bins[NUM_BINS];

        // Assign primitives to bins
        for (int i = task.start; i < task.end; i++) {
          float centroid = tri_centroids[tri_indices[i]][axis];
          int bin_idx = (int)(NUM_BINS * (centroid - centroid_min) /
                              (centroid_max - centroid_min));
          bin_idx = std::min(bin_idx, NUM_BINS - 1);
          bins[bin_idx].Add(tri_bounds[tri_indices[i]]);
        }

        // Compute split positions
        int bins_per_child = (NUM_BINS + num_children - 1) / num_children;
        // int bins_per_child = NUM_BINS / num_children;
        int split_indices[8];

        for (int c = 0; c < num_children; c++) {
          split_indices[c] = (c + 1) * bins_per_child;
        }
        split_indices[num_children - 1] = NUM_BINS;

        // Evaluate cost of this split configuration
        float total_cost = 0.0f;
        int start_bin = 0;

        for (int c = 0; c < num_children; c++) {
          int end_bin = split_indices[c];

          // Merge bins for this child
          AABB child_bounds;
          int child_count = 0;

          for (int b = start_bin; b < end_bin; b++) {
            if (bins[b].count > 0) {
              if (child_count == 0) {
                child_bounds = bins[b].bounds;
              } else {
                child_bounds.Expand(bins[b].bounds);
              }
              child_count += bins[b].count;
            }
          }

          if (child_count > 0) {
            float child_sa = child_bounds.SurfaceArea();
            total_cost += child_sa * child_count * INTERSECTION_COST;
          }

          start_bin = end_bin;
        }

        // Add traversal cost
        total_cost += TRAVERSAL_COST;

        // Normalize by parent surface area
        float parent_sa = node_bounds.SurfaceArea();
        if (parent_sa > 0) {
          total_cost /= parent_sa;
        }

        if (total_cost < best_cost) {
          best_cost = total_cost;
          best_axis = axis;
          best_num_children = num_children;
          for (int c = 0; c < num_children; c++) {
            best_split_indices[c] = split_indices[c];
          }
        }
      }
    }
    // TODO: Support, fornow works without
    // This should try to compare traversal cost VS the intersection cost, in
    // our case with 1 ray vs 8 prims,intersection cost is far lower so it
    // should try to append 8 prims per child
    // float leaf_cost = tri_count * INTERSECTION_COST;
    // if (best_cost >= leaf_cost || best_num_children < 2) {
    //  bvh8_nodes[task.node_index].tri_count = tri_count;
    //  for (int i = 0; i < tri_count; i++)
    //    bvh8_nodes[task.node_index].child_indices[i] = task.start + i;
    //  for (int i = tri_count; i < 8; i++)
    //    bvh8_nodes[task.node_index].child_indices[i] = -1;
    //  continue;
    //}

    float centroid_min = centroid_bounds.min[best_axis];
    float centroid_max = centroid_bounds.max[best_axis];

    // Sort primitives into bins
    std::vector<int> bin_assignments(tri_count);
    for (int i = 0; i < tri_count; i++) {
      float centroid = tri_centroids[tri_indices[task.start + i]][best_axis];
      int bin_idx = (int)(NUM_BINS * (centroid - centroid_min) /
                          (centroid_max - centroid_min));
      bin_idx = std::min(bin_idx, NUM_BINS - 1);
      bin_assignments[i] = bin_idx;
    }

    int bin_to_child[NUM_BINS];
    int start_bin = 0;
    for (int c = 0; c < best_num_children; c++) {
      int end_bin = best_split_indices[c];
      for (int b = start_bin; b < end_bin; b++) {
        bin_to_child[b] = c;
      }
      start_bin = end_bin;
    }

    // Partition tri_indices array in-place
    std::vector<int> child_starts(best_num_children + 1);
    child_starts[0] = task.start;

    std::vector<int> child_counts(best_num_children, 0);
    for (int i = 0; i < tri_count; i++) {
      int child = bin_to_child[bin_assignments[i]];
      child_counts[child]++;
    }

    for (int c = 0; c < best_num_children; c++) {
      child_starts[c + 1] = child_starts[c] + child_counts[c];
    }

    std::vector<int> temp_indices(tri_count);
    std::vector<int> write_positions = child_starts;

    for (int i = 0; i < tri_count; i++) {
      int tri_idx = tri_indices[task.start + i];
      int child = bin_to_child[bin_assignments[i]];
      int write_pos = write_positions[child]++;
      temp_indices[write_pos - task.start] = tri_idx;
    }

    // Copy back
    for (int i = 0; i < tri_count; i++) {
      tri_indices[task.start + i] = temp_indices[i];
    }

    // Reserves space for exactly 8 contiguous children
    int first_child = bvh8_nodes.size();
    bvh8_nodes[task.node_index].tri_count = 0;
    for (int i = 0; i < best_num_children; i++) {
      bvh8_nodes.push_back(BVH8Node::Invalid());
      bvh8_nodes[task.node_index].child_indices[i] = first_child + i;
    }
    for (int i = best_num_children; i < 8; i++)
      bvh8_nodes[task.node_index].child_indices[i] = -1;

    // Compute bounds and queue valid children
    for (int i = 0; i < best_num_children; i++) {
      int child_start = child_starts[i];
      int child_end = child_starts[i + 1];
      int child_count = child_end - child_start;

      if (child_count > 0) {
        AABB child_bounds = tri_bounds[tri_indices[child_start]];
        for (int j = child_start + 1; j < child_end; j++) {
          child_bounds.Expand(tri_bounds[tri_indices[j]]);
        }

        int child_index = first_child + i;
        bvh8_nodes[task.node_index].child_min_x[i] = child_bounds.min.x;
        bvh8_nodes[task.node_index].child_min_y[i] = child_bounds.min.y;
        bvh8_nodes[task.node_index].child_min_z[i] = child_bounds.min.z;
        bvh8_nodes[task.node_index].child_max_x[i] = child_bounds.max.x;
        bvh8_nodes[task.node_index].child_max_y[i] = child_bounds.max.y;
        bvh8_nodes[task.node_index].child_max_z[i] = child_bounds.max.z;

        task_queue.push({child_index, child_start, child_end});
      }
    }
    for (int i = best_num_children; i < 8; i++) {
      bvh8_nodes[task.node_index].child_min_x[i] = FLT_MAX;
      bvh8_nodes[task.node_index].child_min_y[i] = FLT_MAX;
      bvh8_nodes[task.node_index].child_min_z[i] = FLT_MAX;
      bvh8_nodes[task.node_index].child_max_x[i] = -FLT_MAX;
      bvh8_nodes[task.node_index].child_max_y[i] = -FLT_MAX;
      bvh8_nodes[task.node_index].child_max_z[i] = -FLT_MAX;
    }
  }

  return root_index;
}