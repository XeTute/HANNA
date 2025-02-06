# Hello, Friend | ہیلو، دوست | 朋友你好

## What's HANNA
HANNA, short for Hamzah's Artificial Neural Network Architecture, is a headers-only, native C++ library designed for easily inferencing and training HANNA and MLP models.  

> [!IMPORTANT]
> HANNA is still in active development. Please do not currently consider it for production enviroments. See reasons below.

### What HANNA currently does implement:  
- A simple MLP class with SIMD(Single-Instruction Multiple-Data) enabled throught std::valarray + -O3 / compiler optimisations  
- A minimal DPP(Data Pre-Processing) namespace for currently (only) reading CSV files  

### What we plan to implement / currently not implemented:
- The actual HANNA architecture after extensively testing the MLP implementation
- Multiple Threads being used when set to do so (both through std::thread in <thread> for CPUs and OpenCL for GPGPUs)
- Normalisation functions in the DPP namespace

Contributions are welcome.
