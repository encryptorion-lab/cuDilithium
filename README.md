# cuDilithium

## Description

CUDA implementation of Dilithium, a post-quantum signature scheme.

## Prerequisites

All the prerequisites are listed below. You can certainly use hardware and software that is not listed below, but we do not guarantee that the code will work.

### Hardware

- Intel/AMD CPU with 8 performance cores or higher (recommended)
- NVIDIA GPU with compute capability 7.0 or higher (recommended)

### Software

- GCC/G++ 11 or higher (recommended)
- CMake 3.20 or higher (recommended, if not, manually change the CMakeLists.txt)
- CUDA 11.0 or higher (recommended)
- IDE (optional, CLion is recommended)

### Third-party libraries

- OpenMP. Used for CPU multi-threading. Each CUDA stream is assigned to a CPU thread.

## Configuring and Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80-real
cmake --build build -j
```

## Testing and Benchmarking

```bash
./build/bin/test_cuDilithium2
./build/bin/test_cuDilithium3
./build/bin/test_cuDilithium5

./build/bin/bench_cuDilithium2
./build/bin/bench_cuDilithium2
./build/bin/bench_cuDilithium3
./build/bin/bench_cuDilithium5
```

## License

This project (cuDilithium) is released under GPLv3 license. See [COPYING](COPYING) for more information.

Some header files contain the modified code from [Dilithium official repository](https://github.com/pq-crystals/dilithium). These codes are released under Apache 2.0 License. See [Apache 2.0 License](./include/APACHE_LICENSE) for more information.

The CUDA implementation of FIPS202 is released under MIT license. See [MIT License](src/fips202/MIT_LICENSE) for more information.

The FIPS202 reference implementation and the random number generator is released under CC0 license. See [CC0 License](src/fips202/CC0_LICENSE) for more information.
