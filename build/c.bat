echo off
g++ main.cpp "ANNA/ANNA-CPU/ANNA-CPU.hpp" "ANNA/ANNA-CPU/ANNA-CPU.cpp" "ANNA/Data-Prep/CSV.hpp" -O3 -o "build/main.exe"
echo on