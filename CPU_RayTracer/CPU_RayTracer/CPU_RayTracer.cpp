#include "CPU_RayTracer.h"

int main() {
  std::cout << "CPU based Ray Tracing" << std::endl;
  Renderer* renderer = new Renderer();

  renderer->Render("Test1.png", 100, 100);
  return 0;
}
