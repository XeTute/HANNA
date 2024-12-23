#include "ANNA-CPU.hpp"

namespace ANNA_CPU
{
    struct color { std::uint8_t r = 0; std::uint8_t g = 0; std::uint8_t b = 0; };

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
            std::cout << log_buff.str();
            log_buff.str("");
            log_buff.clear();
        }
    }

    ANNA::ANNA()
    {
        layers = 0;
        dLayers = 0;

        scale = std::vector<n>(0);
        neuron = std::vector<NEURON>(0);
        weight = a3d(0);

        color rgb;
        rgb.g = 255;
        rgb.b = 150;
        log("[ANNA-CPU]: Ready to initialize.", rgb, true);
    }
}