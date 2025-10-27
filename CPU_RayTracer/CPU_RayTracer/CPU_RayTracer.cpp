#include "CPU_RayTracer.h"
#include <chrono>

int main() {
  std::cout << "CPU based Ray Tracing" << std::endl;
  Renderer* renderer = new Renderer();

  renderer->Render("Test1.png", 1000, 1000, 1);

  return 0;
}
