# CMakeList.txt : CMake project for CPU_RayTracer, include source and define
# project specific logic here.

find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

add_subdirectory ("src")
add_subdirectory ("includes")

add_executable (CPU_RayTracer ${SOURCES} ${include_files})

target_include_directories(CPU_RayTracer PRIVATE source_dir, include_dir)
target_link_libraries(CPU_RayTracer PRIVATE glm::glm)
  target_link_libraries(CPU_RayTracer PRIVATE assimp::assimp)

  
# target_compile_definitions(CPU_RayTracer PUBLIC DEBUG_MODEL)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CPU_RayTracer PROPERTY CXX_STANDARD 20)
endif()