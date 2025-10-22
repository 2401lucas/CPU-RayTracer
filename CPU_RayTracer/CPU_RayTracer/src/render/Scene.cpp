#include "Scene.h"

#include <iostream>

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
  spheres.count = 2;

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

  LoadModel("assets/Cube/Cube.gltf", {0, 0, 5});
}

// Loads a model at filepath into the primary vertex buffers
void Scene::LoadModel(const char* filepath, aiVector3f offset) {
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(filepath, aiProcess_Triangulate);
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

  for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
    auto& mesh = scene->mMeshes[i];
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
}

void Scene::CalculateBVH() {
  // Calculate AABB
  // Create Root
  // Split nodes (SAH)
  // Create child nodes
  // Store Hierarchy
  // Flatten
}
