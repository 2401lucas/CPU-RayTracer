#pragma once
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <glm/glm.hpp>

class Scene {
 public:
  Scene(int max_spheres, int max_faces);
  ~Scene();

  void LoadScene(const char* filepath);
  void LoadModel(const char* filepath, aiVector3f offset = {0,0,0});
  void CalculateBVH();

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
};