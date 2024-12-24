#include <cmath>
#include <cstdint>

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

template <typename T>
T softmax(T z_i, T e_z_sum) { return std::exp(z_i) / e_z_sum; } // This assumes that the second param is passed in as std::exp(z_sum) to save z_n calculations

template <typename T>
T sum(T* array, n size)
{
    T sum(0);
    for (n i = 0; i < size; ++i) sum += array[i];
    return sum;
}

template <typename T>
T partialSum(T* array, n start, n end)
{
    T sum(0);
    for (n i = start; i < end; ++i) sum += array[i];
    return sum;
}