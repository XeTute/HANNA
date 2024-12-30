@echo off
g++ main.cpp "ANNA/ANNA-CPU/ANNA-CPU.hpp" "ANNA/ANNA-CPU/ANNA-CPU.cpp" "ANNA/Data-Prep/CSV.hpp" "ANNA/ANNA-CPU/Math-Fun.hpp" "ANNA/ANNA-CPU/MLP-Layer.hpp" -O3 -o "build/main.exe"
build\main.exe
@echo on