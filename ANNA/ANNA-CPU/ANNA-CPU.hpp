#ifndef ANNA_CPU_HPP
#define ANNA_CPU_HPP

#include "MLP-Layer.hpp"
#include <string>
#include <fstream>
#include <sstream>

namespace ANNA_CPU
{
    class ANNA
    {
    private:

        std::vector<MLP::LAYER> MLPL;
        std::vector<std::vector<float>> tmp_layer;
        std::vector<n> scale;
        std::vector<n> ds; // decreased scale, scale without the first element for more effective hidden & output layer construction

        n layers;
        n dl; // decreased layers
        n ddl;

        unsigned short threads;

        void basicInit(std::vector<n> neurons, void (*_activation) (float&))
        {
            scale = neurons;

            layers = neurons.size();
            dl = layers - 1;
            ddl = layers - 2;
            activation = _activation;

            ds = neurons;
            ds.erase(ds.begin());

            MLPL.resize(dl);
        }

    public:

        void (*activation)(float&);
        void (*activationDV)(float&);

        float lr = 0.1f;

        ANNA() : MLPL(0), tmp_layer(0), scale(0), ds(0), layers(0), dl(0), ddl(0), threads(1) {};
        ANNA(std::vector<n> neurons, void (*_activation)(float&))
        {
            basicInit(neurons, _activation);

            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l], false);
                MLPL[l].rand();
            }
        }

        ANNA(std::vector<n> neurons, void (*_activation)(float&), void (*_activationDV)(float&))
        {
            basicInit(neurons, _activation);

            activationDV = _activationDV;
            tmp_layer.resize(dl);

            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l], true);
                MLPL[l].rand();
            }
        }

        bool save(std::string path)
        {
            std::ofstream w(path, std::ios::out | std::ios::binary);
            if (!w.is_open() || !w) return false;

            for (n neurons : scale) w << neurons << ';';
            w << '\n';

            for (n l = 0; l < dl; ++l)
            {
                w.write(reinterpret_cast<const char*>(MLPL[l].getBias().data()), MLPL[l].getBias().size() * sizeof(float));
                for (std::vector<float> neuron : MLPL[l].getWeights())
                    w.write(reinterpret_cast<const char*>(neuron.data()), neuron.size() * sizeof(float));
            }

            w.close();
            return true;
        }

        bool load(std::string path, void (*_activation)(float&))
        {
            std::ifstream r(path, std::ios::in | std::ios::binary);
            if (!r.is_open() || !r) return false;
            r.seekg(0);

            scale = std::vector<n>(0);

            {
                std::string strBuffer("");
                std::stringstream ssBuffer("");
                char charBuffer = 0;

                std::getline(r, strBuffer);
                ssBuffer = std::stringstream(strBuffer);
                
                while (std::getline(ssBuffer, strBuffer, ';'))
                    scale.push_back(std::stoull(strBuffer));
            }
            basicInit(scale, activation);

            std::vector<n> mlp_scale(scale);
            mlp_scale.erase(mlp_scale.begin());

            for (n l = 0; l < dl; ++l)
            {
                std::vector<float> bias(mlp_scale[l], 0.f);
                std::vector<std::vector<float>> weight_ll(mlp_scale[l], std::vector<float>(scale[l], 0.f));

                if (!r.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float)))
                {
                    std::cerr << "Failed to read: std::ifstream returned false while trying to read binary. ANNA may be corrupted.\n";
                    return 0;
                }

                for (std::vector<float> neuron : weight_ll)
                {
                    if (!r.read(reinterpret_cast<char*>(neuron.data()), neuron.size() * sizeof(float)))
                    {
                        std::cerr << "Failed to read: std::ifstream returned false while trying to read binary. ANNA may be corrupted.\n";
                        return 0;
                    }
                }

                MLPL[l].pretrained(bias, weight_ll, true);
            }

            return true;
        }

        void display()
        {
            std::cout << "\n\nDISPLAY START";
            for (n l = 0; l < dl; ++l)
            {
                std::cout << "\n<--- LAYER " << l << " --->\n";
                std::cout << "Neuron-Count: " << MLPL[l].getState().size() << '\n';
                std::cout << "Bias: \n";
                for (float bias : MLPL[l].getBias()) std::cout << "- " << bias << '\n';

                std::cout << "\nWeights: \n";
                for (n neuron = 0; neuron < MLPL[l].getBias().size(); ++neuron)
                {
                    std::cout << ">-- NEURON " << neuron << " <--\n";
                    for (n nll = 0; nll < MLPL[l].getWeights()[0].size(); ++nll)
                        std::cout << "- " << MLPL[l].getWeights()[neuron][nll] << '\n';
                }
            }
            std::cout << "DISPLAY END\n\n";
        }

        void setThreads(unsigned short _threads) { threads = _threads; }
        unsigned short getThreads() { return threads; }

        void forward(const std::vector<float>& inp)
        {
            MLPL[0].forward(inp, activation);
            for (n l = 1; l < dl; ++l)
                MLPL[l].forward(MLPL[l - 1].getState(), activation);
        }

        void forwardForGrad(const std::vector<float>& inp)
        {
            tmp_layer[0] = inp;
            MLPL[0].forward(inp, activation);
            for (n l = 1; l < dl; ++l)
                MLPL[l].forward(MLPL[l - 1].getState(), activation);
        }

        void gradDesc(const std::vector<float>& expected_output)
        {
            tmp_layer[ddl] = MLPL[ddl].gradDesc(expected_output, MLPL[ddl - 1].getState(), activationDV, lr);
            for (n l = (ddl - 1); l > 0; --l)
                tmp_layer[l] = MLPL[l].gradDesc(tmp_layer[l + 1], MLPL[l - 1].getState(), activationDV, lr);
            MLPL[0].gradDesc(tmp_layer[1], tmp_layer[0], activationDV, lr);
        }

        const std::vector<float>& getOutput() { return MLPL[ddl].getState(); }
        void calcSoftmaxOut() { MLPL[ddl].softmax(); }

        void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, n EPOCHS)
        {
            if (input.size() != output.size())
            {
                std::cerr << "[ANNA-CPU train(...)]: The sizes of input & output do not match. Won't train.\n";
                return;
            }

            n samples = input.size();

            if (threads == 1)
            {
                for (n e = 0; e < EPOCHS; ++e)
                {
                    for (n s = 0; s < samples; ++s)
                    {
                        this->forwardForGrad(input[s]);
                        this->gradDesc(output[s]);
                    }
                }
                return;
            }

            n chunkSize = std::ceil(samples / threads);
            n remainder = samples % threads;

            std::vector<ANNA> copy(threads);
        }
    };
}

#endif