#ifndef MATH_FUN_HPP
#define MATH_FUN_HPP

#include <cmath>
#include <cstdint>

namespace MATH
{
    typedef uint64_t n;

    template <typename T>
    T sigmoid(T x) { return T(1) / (1 + std::exp(-1)); }
    template <typename T>
    T sigmoidDv(T x)
    {
        T sigm = sigmoid(x);
        return sigm * (1 - sigm);
    }

    template <typename T>
    T ReLU(T x) { return (x > T(0)) ? x : T(0); }
    template <typename T>
    T ReLUDv(T x) { return x; }

    // This assumes that the second param is passed in as std::exp(z_sum) to save z_n calculations
    template <typename T>
    T softmax(T z_i, T e_z_sum) { return std::exp(z_i) / e_z_sum; }

    template <typename T>
    void softmax(T& result, T& z_i, T e_z_sum) { z_i = std::exp(z_i) / e_z_sum; }
}

#endif