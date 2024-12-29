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
    struct normConf { T x, y; };

    template <typename T>
    normConf<T> minMaxNorm(std::vector<T>& data)
    {
        normConf<T> config;
        config.x = *std::min_element(data.begin(), data.end()); // min val
        config.y = *std::max_element(data.begin(), data.end()) - config.x; // delta
        n ne = data.size(); // n elems
        
        for (n i = 0; i < ne; ++i) data[i] = T(data[i] - config.x) / config.y;
        return config;
    }

    template <typename T>
    void minMaxNorm(std::vector<T>& data, const normConf<T>& config)
    {
        n ne = data.size();
        for (n i = 0; i < ne; ++i)
            data[i] = T(data[i] - config.x) / config.y;
    }

    template <typename T>
    normConf<T> ZScaleNorm(std::vector<T>& data)
    {
        normConf<T> config;
        n ne = data.size(); // n elems
        config.x = std::accumulate(data.begin(), data.end(), T(0)) / ne;
        config.y = std::sqrt
        (
            std::inner_product(data.begin(), data.end(), data.begin(), T(0))
            / ne - config.x * config.x
        );

        for (n i = 0; i < ne; ++i) data[i] = T(data[i] - config.x) / config.y;
        return config;
    }

    template <typename T>
    void ZScaleNorm(std::vector<T>& data, const normConf<T>& config)
    {
        n ne = data.size();
        for (n i = 0; i < ne; ++i) data[i] = T(data[i] - config.x) / config.y;
    }
}

#endif