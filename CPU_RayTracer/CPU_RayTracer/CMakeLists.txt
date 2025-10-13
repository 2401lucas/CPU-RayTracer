# CMakeList.txt : CMake project for CPU_RayTracer, include source and define
# project specific logic here.

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

add_subdirectory ("src")
add_subdirectory ("includes")

add_executable (CPU_RayTracer ${SOURCES} "CPU_RayTracer.cpp" "CPU_RayTracer.h")

target_include_directories(CPU_RayTracer PRIVATE source_dir, include_dir)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CPU_RayTracer PROPERTY CXX_STANDARD 20)
endif()