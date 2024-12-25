#ifndef MLP_LAYER_HPP
#define MLP_LAYER_HPP

#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#include "Math-Fun.hpp"

using n = uint64_t;

namespace MLP
{
    class LAYER
    {
    private:

        std::vector<float> val;
        std::vector<float> bias;
        std::vector<std::vector<float>> weight_ll;

        n neurons;
        n neurons_in;

    public:

        LAYER(): neurons(0) {}

        void create(n Neurons, n last_layer)
        {
            neurons = Neurons;
            val.resize(neurons);
            bias.resize(neurons);
            weight_ll.resize(neurons);

            neurons_in = last_layer;

            if (last_layer > 0)
            {
                for (std::vector<float>& neuron : weight_ll)
                    neuron.resize(last_layer, 0.f);
            }
        }

        void setRand()
        {
            float limit = sqrtf(6.f / neurons_in);
            std::uniform_real_distribution<float> dist(-limit, limit);
            std::mt19937 mt(std::random_device{}());

            for (std::vector<float>& neuron : weight_ll)
                for (float& x : neuron) x = dist(mt);
            for (float x : bias) x = dist(mt);
        }

        void forward(const std::vector<float>& input, void (*activation)(float&))
        {
            for (n neuron = 0; neuron < neurons; ++neuron)
            {
                val[neuron] = bias[neuron];
                for (n i = 0; i < neurons_in; ++i) val[neuron] += input[i] * weight_ll[neuron][i];
                activation(val[neuron]);
            }
        }

        void softmax()
        {
            float sum = 0.f;
            for (float& x : val) sum += expf(x); // https://stackoverflow.com/questions/55458487/stdexpf-and-stdlogf-not-recognized-by-gcc-7-2-0
            for (float& x : val) MATH::softmax(x, x, sum);
        }

        const std::vector<float>& getState() const { return val; }
    };
}

#endif