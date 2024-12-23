#ifndef ANNA_CPU_HPP
#define ANNA_CPU_HPP
#endif

#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <mutex>
#include <vector>

#include "Math-Fun.cpp"

typedef uint64_t n;
typedef float prec;

typedef std::vector<prec> a1d; // array 1 dim
typedef std::vector<a1d> a2d;
typedef std::vector<a2d> a3d;

namespace ANNA_CPU
{
    struct NEURON
    {
        prec val;
        prec bias;
    };

    class ANNA
    {
    private:

        n layers;
        n dLayers; // decreased layers

        std::vector<n> scale;
        std::vector<NEURON> neuron;
        a3d weight;

    public:

        float lr = 0.1f;

        ANNA();

    };
};