> ANNA is still new, and it may have some small bugs or undefined behaviour which we may did not expirience while testing. Please test your application before publishing and issue bugs when they happen.

# ANNA
ANNA(Asadullah's Neural Network Architecture) is our new experimental architecture which we actively develop.
It's supposed to be a more efficient alternative to the Transformers architecture for roleplaying, story-telling and similar tasks which don't require the amount of performance as the average Transformer model offers.
Currently, it's simply an implementation of a MLP(Multi-Layer-Perceptron) with thread support for large datasets with minimal to no overhead using the ANNA::train function. As far as we know, SIMD is automatically applied to std::vector for compilers like GCC and MSVC.

# Getting started
Ensure you have `git` and `g++` installed on the device you want to run following commands on.
First,
```
git clone https://github.com/XeTute/ANNA
cd ANNA
```
then, on Linux (NOT Termux, it's version of g++ isn't really doing what it's supposed to), you can run `car.sh`(compile and run), or
```
g++ -D__USE_MINGW_ANSI_STDIO=0 -O3 -Ofast -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -ffast-math -fwhole-program -fno-exceptions -fno-rtti -fexceptions -pthread main.cpp ANNA.hpp -o anna
chmod +x anna
./anna
```
or, if you're on Windows, you can run `car.bat`, or
```
g++ -D__USE_MINGW_ANSI_STDIO=0 -O3 -Ofast -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -ffast-math -fwhole-program -fno-exceptions -fno-rtti -fexceptions -mthread main.cpp ANNA.hpp -o anna.exe
anna.exe
```
This will download, compile and execute the latest main.cpp found on this GitHub repo.

# Other Notices
Currently, only CPUs(with thread support on ANNA::train for large datasets) are supported, but we're planning to add support for CUDA 12.6.

# Star History
<a href="https://star-history.com/#XeTute/ANNA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
 </picture>
</a>
