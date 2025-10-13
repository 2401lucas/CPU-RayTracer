#include "Renderer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../includes/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../includes/stb_image_write.h"

Renderer::Renderer() {}

Renderer::~Renderer() {}

void Renderer::Render(const char* filename, const uint32_t width,
                      const uint32_t height) {
  constexpr uint8_t CHANNEL_NUM = 3;
  uint8_t* pixels = new uint8_t[width * height * CHANNEL_NUM];

  int index = 0;
  for (int j = height - 1; j >= 0; --j) {
    for (int i = 0; i < width; ++i) {
      float r = (float)i / (float)width;
      float g = (float)j / (float)height;
      float b = 0.2f;
      uint8_t ir = uint8_t(255.0 * r);
      uint8_t ig = uint8_t(255.0 * g);
      uint8_t ib = uint8_t(255.0 * b);
      pixels[index++] = ir;
      pixels[index++] = ig;
      pixels[index++] = ib;
    }
  }
  stbi_write_png(filename, width, height, CHANNEL_NUM, pixels,
                 width * CHANNEL_NUM);
}
