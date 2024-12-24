#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include <iostream>
#include <cstdint>
#include <sstream>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace normalize
{
    typedef uint64_t n;

    struct color { uint8_t r = 0; uint8_t g = 0; uint8_t b = 150; };

    std::ostringstream log_buff("");
    std::mutex log_buff_mutex;
    void log(std::string str, color rgb, bool display)
    {
        {
            std::lock_guard<std::mutex> guard(log_buff_mutex);
            log_buff << "\033[38;2;" + std::to_string(rgb.r) + ";" + std::to_string(rgb.g) + ";" + std::to_string(rgb.b) + "m" + str + "\033[0m\n";
        }
        if (display)
        {
            std::cerr << log_buff.str();
            log_buff.str("");
            log_buff.clear();
        }
    }

    template <typename T>
    std::vector<T> minMaxNorm(const std::vector<T>& data)
    {
        T mnv = *std::min_element(data.begin(), data.end()); // min val
        T mxv = *std::max_element(data.begin(), data.end());
        T d = mxv - mnv; // delta
        n ne = data.size(); // n elems

        std::vector<T> norm(ne, T(0)); // normalized data
        for (n i = 0; i < ne; ++i) norm[i] = T(data[i] - mnv) / d;
        return norm;
    }

    template <typename T>
    void minMaxNorm(std::vector<T>& data)
    {
        T mnv = *std::min_element(data.begin(), data.end()); // min val
        T mxv = *std::max_element(data.begin(), data.end());
        T d = mxv - mnv; // delta
        n ne = data.size(); // n elems
        
        for (n i = 0; i < ne; ++i) data[i] = T(data[i] - mnv) / d;
    }

    template <typename T>
    std::vector<T> ZScaleNorm(const std::vector<T>& data)
    {
        n ne = data.size(); // n elems
        T mean = std::accumulate(data.begin(), data.end(), T(0)) / ne;
        T stddev = std::sqrt
        (
            std::inner_product(data.begin(), data.end(), data.begin(), T(0))
            / ne - mean * mean
        );

        std::vector<T> norm(ne); // normalized data
        for (n i = 0; i < ne; ++i) norm[i] = T(data[i] - mean) / stddev;
        return norm;
    }

    template <typename T>
    void ZScaleNorm(std::vector<T>& data)
    {
        n ne = data.size(); // n elems
        T mean = std::accumulate(data.begin(), data.end(), T(0)) / ne;
        T stddev = std::sqrt
        (
            std::inner_product(data.begin(), data.end(), data.begin(), T(0))
            / ne - mean * mean
        );

        for (n i = 0; i < ne; ++i) data[i] = T(data[i] - mean) / stddev;
    }
}

#endif