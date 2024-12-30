#ifndef MATH_FUN_HPP
#define MATH_FUN_HPP

#include <cmath>
#include <cstdint>

namespace MATH
{
    typedef uint64_t n;

    template <typename T>
    void sigmoid(T& x) { x = T(1) / (1 + std::exp(-x)); }
    template <typename T>
    void sigmoidDv(T& x) { x *= (1 - x); }

    template <typename T>
    void ReLU(T& x) { x = (x > T(0)) ? x : T(0); }
    template <typename T>
    void ReLUDv(T& x) { return (x >= 0) ? T(1) : T(0); }

    // This assumes that the second param is passed in as std::exp(z_sum) to save z_n calculations
    template <typename T>
    T softmax(T z_i, T e_z_sum) { return std::exp(z_i) / e_z_sum; }

    template <typename T>
    void softmax(T& result, T& z_i, T e_z_sum) { z_i = std::exp(z_i) / e_z_sum; }
}

#endif