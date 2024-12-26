#ifndef MLP_LAYER_HPP
#define MLP_LAYER_HPP

#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

#include "Math-Fun.hpp"

using n = uint64_t;

namespace MLP
{
    class LAYER
    {
    private:

        std::vector<float> val;
        std::vector<float> bias;
        std::vector<float> expct_inp;

        std::vector<std::vector<float>> weight_ll;

        n neurons;
        n neurons_in;

        bool train;

    public:

        LAYER(): val(0), bias(0), expct_inp(0), weight_ll(0), neurons(0), neurons_in(0), train(false) {}

        void create(n Neurons, n last_layer, bool _train)
        {
            neurons = Neurons;
            neurons_in = last_layer;
            train = _train;

            val.resize(neurons);
            bias.resize(neurons);
            weight_ll = std::vector<std::vector<float>>(neurons, std::vector<float>(last_layer, 0.f));
            
            if (train) expct_inp.resize(neurons_in);
        }

        void rand()
        {
            std::uniform_real_distribution<float> dist(-5.f, 5.f);
            std::mt19937 mt(std::random_device{}());

            for (std::vector<float>& neuron : weight_ll)
                for (float& x : neuron) x = dist(mt);
            for (float x : bias) x = dist(mt);
        }

        void pretrained(std::vector<float> _bias, std::vector<std::vector<float>> _weight_ll, bool train)
        {
            bias = _bias;
            weight_ll = _weight_ll;

            neurons = bias.size();
            val.reserve(bias.size());

            if (weight_ll.size() > 0) neurons_in = weight_ll[0].size();
            else neurons_in = 0;

            if (train) expct_inp.resize(neurons_in);
        }

        void forward(const std::vector<float>& input, void (*activation)(float&))
        {
            for (n neuron = 0; neuron < neurons; ++neuron)
            {
                val[neuron] = bias[neuron];
                for (n i = 0; i < neurons_in; ++i)
                    val[neuron] += input[i] * weight_ll[neuron][i];
                activation(val[neuron]);
            }
        }

        const std::vector<float>& gradDesc(const std::vector<float>& expctd_out, const std::vector<float>& input, void (*activation_dr)(float&), float& lr)
        {
            for (float& x : expct_inp) x = 0.f;
            for (n neuron = 0; neuron < neurons; ++neuron)
            {
                float delta = expctd_out[neuron] - val[neuron];
                activation_dr(val[neuron]);
                delta *= val[neuron];

                bias[neuron] += delta * lr;
                for (n neuron_ll = 0; neuron_ll < neurons_in; ++neuron_ll)
                {
                    weight_ll[neuron][neuron_ll] += input[neuron_ll] * delta * lr;
                    expct_inp[neuron_ll] += delta * weight_ll[neuron][neuron_ll];
                }
            }

            return expct_inp;
        }

        void softmax()
        {
            float sum = 0.f;
            for (float& x : val) sum += expf(x); // https://stackoverflow.com/questions/55458487/stdexpf-and-stdlogf-not-recognized-by-gcc-7-2-0
            for (float& x : val) MATH::softmax(x, x, sum);
        }

        const std::vector<std::vector<float>>& getWeights() const { return weight_ll; }
        const std::vector<float>& getBias() const { return bias; }

        const std::vector<float>& getState() const { return val; }

        ~LAYER()
        {
            bias.clear();
            val.clear();
            weight_ll.clear();
            expct_inp.clear();
            
            neurons = 0;
            neurons_in = 0;
        }
    };
}

#endif