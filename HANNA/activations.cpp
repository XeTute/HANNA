#include <cmath>

#ifndef ACTIVATIONS_CPP
#define ACTIVATIONS_CPP

constexpr float leaky_ReLU_slope = 0.05f;

template <typename dt>
dt sigmoid(const dt& x) { return dt(1) / dt(1.f + std::exp(float(-x))); }
template <typename dt>
dt sigmoid_derivative(const dt& x) { dt s = sigmoid(x); return s * (dt(1) - s); }

template <typename dt>
dt ReLU(const dt& x) { return (x > dt(0)) ? x : dt(0); }
template <typename dt>
dt ReLU_derivative(const dt& x) { return (x > dt(0)) ? dt(1) : dt(0); }

template <typename dt>
dt leaky_ReLU(const dt& x) { return (x > dt(0)) ? x : x * dt(leaky_ReLU_slope); }
template <typename dt>
dt leaky_ReLU_derivative(const dt& x) { return (x > dt(0)) ? dt(1) : dt(leaky_ReLU_slope); }

#endif
