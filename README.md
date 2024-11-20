# ANNA
ANNA(Asadullah's Neural Network Architecture) is our new experimental architecture which we actively develop.
It's supposed to be a more efficient alternative to the Transformers architecture for roleplaying, story-telling and similar tasks which don't require the amount of performance as the average Transformer model offers.
Currently, it's simply an implementation of a MLP(Multi-Layer-Perceptron) with thread support through OMP.

# Getting started
Ensure you have `git` and `g++` installed on the device you want to run following commands on.
First,
```
git clone https://github.com/XeTute/ANNA
cd ANNA
```
then, on Linux, you can run `car.sh`(compile and run), or
```
g++ -D__USE_MINGW_ANSI_STDIO=0 -O3 -Ofast -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -ffast-math -fopenmp -fwhole-program -fno-exceptions -fno-rtti -fexceptions main.cpp ANNA.hpp -o anna
chmod +x anna
./anna
```
or, if you're on Windows, you can run `car.bat`, or
```
g++ -D__USE_MINGW_ANSI_STDIO=0 -O3 -Ofast -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -ffast-math -fopenmp -fwhole-program -fno-exceptions -fno-rtti -fexceptions main.cpp ANNA.hpp -o anna.exe
anna.exe
```
This will download, compile and execute the latest main.cpp found on this GitHub repo.

# Other Notices
Currently, only CPUs(which support OpenMP / OMP) are supported, but we're planning to add support for CUDA.

Performance-wise, we noticed that smaller models(less than ~1k params) will train significantly faster when being single-threaded.
For models which are larger, it makes sense to train and inference using more than one thread, while keeping the amount of threads used less than the number of cores in your processor.

Compiler-wise, we've expirienced issues on both MinGW(incompatible, Windows thinks that it generated a 16bit?), and MVSC(vector access violations when there are none).
If you notice any issues in regards to ANNA.hpp, please issue it.

# Star History
<a href="https://star-history.com/#XeTute/ANNA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
 </picture>
</a>
