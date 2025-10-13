#include <string>

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Render(const char* filename, const uint32_t width,
              const uint32_t height);
 private:
};