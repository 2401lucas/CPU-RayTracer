# Ray Tracer
Currently the ray tracer is a partial setup of a feature complete ray tracer. It uses single hit tracers to derive colors from a mock scene setup. It is single threaded although the current structure of processing rays 
is adaptable to multithreading and path tracing.
## Key Features
- SIMD
- .gltf Model loading
- Triangle intersection with the [Möller–Trumbore intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)

## TODO
- BVH
- Multithreading
- Bounces
- Lighting
- Interface to modify renders (Reso, depth, samples, scene...)
