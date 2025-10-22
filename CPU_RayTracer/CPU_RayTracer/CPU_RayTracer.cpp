#include "CPU_RayTracer.h"
#include <chrono>

int main() {
  std::cout << "CPU based Ray Tracing" << std::endl;
  Renderer* renderer = new Renderer();

  
  auto start = std::chrono::high_resolution_clock::now();
  renderer->Render("Test1.png", 100, 100, 10);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Render took " << elapsed.count() << " seconds." << std::endl;

  return 0;
}
