#include "CPU_RayTracer.h"
#include <chrono>

int main() {
  std::cout << "CPU based Ray Tracing" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  Renderer* renderer = new Renderer();
  renderer->Render("Test1.png", 1000, 1000, 10);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Render took " << elapsed.count() << " seconds." << std::endl;

  return 0;
}
