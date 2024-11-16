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
then, on Linux:
```
g++ main.cpp ANNA.hpp -O3 -o anna && chmod +x anna && ./anna
```
or, if you're on Windows:
```
g++ main.cpp ANNA.hpp -O3 -o anna.exe
./anna.exe
```
This will download, compile and execute the latest main.cpp found on this GitHub repo.

# Star History
<a href="https://star-history.com/#XeTute/ANNA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
 </picture>
</a>
