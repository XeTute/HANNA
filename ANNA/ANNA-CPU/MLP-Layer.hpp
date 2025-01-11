#ifndef MLP_LAYER_HPP
#define MLP_LAYER_HPP

#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include <thread>

#include "Math-Fun.hpp"

using n = size_t;

namespace MLP
{
    class LAYER
    {
    private:

        std::vector<float> val;
        std::vector<float> bias;
        std::vector<float> expct_inp;

        std::vector<std::vector<float>> weight_ll;

        n neurons, neurons_in;
        n threads, iterations, iremainder;

        void forwardNeuron(const std::vector<float>& input, void (*activation) (float&), n neuron)
        {
            float x = bias[neuron];
            for (n i = 0; i < neurons_in; ++i)
                x += input[i] * weight_ll[neuron][i];
            activation(x);
            val[neuron] = x;
        }

    public:

        LAYER(): val(0), bias(0), expct_inp(0), weight_ll(0), neurons(0), neurons_in(0), threads(1), iterations(0), iremainder(0) {}

        void create(n Neurons, n last_layer)
        {
            neurons = Neurons;
            neurons_in = last_layer;

            val.resize(neurons);
            bias.resize(neurons);
            weight_ll = std::vector<std::vector<float>>(neurons, std::vector<float>(last_layer, 0.f));
        }

        void enableTraining() { expct_inp.resize(neurons_in); }
        void disableTraining() { expct_inp.resize(0); }

        void rand()
        {
            std::uniform_real_distribution<float> dist(-5.f, 5.f);
            std::mt19937 mt(std::random_device{}());

            for (std::vector<float>& neuron : weight_ll)
                for (float& x : neuron) x = dist(mt);
            for (float& x : bias) x = dist(mt);
        }

        void pretrained(std::vector<float> _bias, std::vector<std::vector<float>> _weight_ll)
        {
            bias = _bias;
            weight_ll = _weight_ll;

            neurons = bias.size();
            val.reserve(bias.size());

            if (weight_ll.size() > 0) neurons_in = weight_ll[0].size();
            else neurons_in = 0;
        }

        void setThreads(n& _threads)
        {
            threads = _threads;
            iterations = n(std::ceil(double(neurons / threads)));
            iremainder = n(neurons - (iterations * threads));
        }

        void forward(const std::vector<float>& input, void (*activation)(float&), n& _threads)
        {
            std::vector<std::thread> tp(_threads);

            n neuron = 0;
            for (n i = 0; i < iterations; ++i)
            {
                for (n t = 0; t < threads; ++t)
                {
                    tp[t] = std::thread(&LAYER::forwardNeuron, this, input, activation, neuron);
                    ++neuron;
                }
                for (n t = 0; t < threads; ++t) if (tp[t].joinable()) tp[t].join();
            }

            for (n t = 0; t < iremainder; ++t)
            {
                tp[t] = std::thread(&LAYER::forwardNeuron, this, input, activation, neuron);
                ++neuron;
            }
            for (n t = 0; t < iremainder; ++t) if (tp[t].joinable()) tp[t].join();
        }

        const std::vector<float>& gradDesc(const std::vector<float>& expctd_out, const std::vector<float>& input, void (*activation_dr)(float&), float& lr)
        {
            for (float& x : expct_inp) x = 0.f;
            for (n neuron = 0; neuron < neurons; ++neuron)
            {
                float _val = val[neuron];
                float delta = expctd_out[neuron] - _val;
                activation_dr(_val);
                delta *= _val;

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

        std::vector<std::vector<float>>& getWeights() { return weight_ll; }
        std::vector<float>& getBias() { return bias; }

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

        void operator= (const LAYER& other)
        {
            if (this != &other)
            {
                this->val = other.val;
                this->bias = other.bias;
                this->expct_inp = other.expct_inp;

                this->weight_ll = other.weight_ll;

                this->neurons = other.neurons;
                this->neurons_in = other.neurons_in;
            }
        }
    };
}

#endif