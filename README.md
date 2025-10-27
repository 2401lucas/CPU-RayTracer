# High-Performance CPU Raytracer
A from-scratch CPU-based raytracer built in C/C++ featuring SIMD optimization, parallel rendering with work-stealing multithreading and efficient BVH8 acceleration structures for real-time capable rendering of 3D scenes. Built in about a week to take a deeper dive in CPU optimizations, ray tracing and multi-threading.

# Performance Breakdown
Rendering a simple scene at 1000x1000 resolution with 100 rays per pixel.
10 million Rays, 12 triangles. 

Single Threaded, No BVH - 436.195s

Single Threaded, BVH    - 44.156s (10x faster)

Multi Threaded, BVH     - 0.526s  (829x faster overall)

Note: This simple 12-triangle scene somewhat understates BVH benefits, which scale better with higher triangle counts
### Tested on a Ryzen 7 2700, 8 cores 16 threads

# Key Performance Optimizations
### SIMD
SIMD Vectorization using AVX intrinsics for parallel ray-triangle intersection tests and BVH traversal. With AVX(__m256) registers, it supports single ray traversal against up to 8 bounding boxes or primitives.

### BLAS BVH
Up to 8-way BVH(bounding volume hierarchy) speeding up ray intersections. Each model is processed into a BLAS(Bottom Level Acceleration Structure). The BVH width varies from 2-8 child nodes depending on triangle density. This is to avoid 8 nodes with 1 or 2 triangles each since the cost to intersect fewer than 8 triangles is nearly identical thanks to SIMD.

### Multithreading
- Multithreading with a custom work stealing implementation supporting batched jobs. This is done by enqueueing work into each thread and if a thread runs out of work, it will check other threads work queue and will "steal" their work until all the work is completed. This is particularly useful for Ray Tracing as rays with an early exit condition, such as a miss, have less work to do. [Read more about the implementation here](https://github.com/2401lucas/CPP-MT-JobStealing)

### Other Key Features
- .gltf Model loading
- Triangle intersection with the [Möller–Trumbore intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)


## Missing Features I would like to add upon revisiting
- Tiled Rendering for higher resolutions
- TLAS
- Bounces/indirect lighting - In the current multithreading implementation, bounces would be trivial to add as we could simply enqueue the new rays
- Direct lighting & shadows  
- Texture sampling
- Interface to modify renders(Resolution, depth, samples, scene...)


# Final Thoughts
This week long project achieved its goals, giving me experience with SIMD vectorization, work-stealing schedulers, and ray tracing fundamentals. While still missing features like lighting and bounces, the ~830x performance improvement validates the optimization approach. My next step will be translating these learnings to GPU compute shaders for real-time pathtracing.


# Resources
https://en.algorithmica.org/hpc/ 

This project was inspired from reading this book, I wanted to experiment with some of what I learned. This is far and away my favorite book related to computers; a truly fantastic resource.
