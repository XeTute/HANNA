# ANNA
ANNA(Asadullah's Neural Network Architecture) is our new experimental architecture which we actively develop.
It's supposed to be a more efficient alternative to the Transformers architecture for roleplaying, story-telling and similar tasks which don't require the amount of performance as the average Transformer model offers.
Currently, it's simply an implementation of a MLP(Multi-Layer-Perceptron) with thread support through std::async coming soon.

# Getting started
Ensure you have `git` and `g++` installed on the device you want to run following commands on.
First,
```
git clone https://github.com/XeTute/ANNA
cd ANNA
```
then, on Linux (NOT Termux, it's version of g++ isn't really doing what it's supposed to), you can run `car.sh`(compile and run), or
```
g++ -D__USE_MINGW_ANSI_STDIO=0 -O3 -Ofast -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -ffast-math -fwhole-program -fno-exceptions -fno-rtti -fexceptions main.cpp ANNA.hpp -o anna
chmod +x anna
./anna
```
or, if you're on Windows, you can run `car.bat`, or
```
g++ -D__USE_MINGW_ANSI_STDIO=0 -O3 -Ofast -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -ffast-math -fwhole-program -fno-exceptions -fno-rtti -fexceptions main.cpp ANNA.hpp -o anna.exe
anna.exe
```
This will download, compile and execute the latest main.cpp found on this GitHub repo.

# Other Notices
Currently, only CPUs(std::async support coming soon) are supported, but we're planning to add support for CUDA.

# Star History
<a href="https://star-history.com/#XeTute/ANNA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
 </picture>
</a>
