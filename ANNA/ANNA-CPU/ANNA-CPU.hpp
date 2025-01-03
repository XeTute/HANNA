#ifndef ANNA_CPU_HPP
#define ANNA_CPU_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <iostream>
#include <algorithm>

#include "MLP-Layer.hpp"

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
        bool saved = false;

        void basicInit(std::vector<n> neurons, void (*_activation) (float&))
        {
            scale = neurons;

            layers = neurons.size();
            dl = layers - 1;
            ddl = layers - 2;
            activation = _activation;

            ds = neurons;
            ds.erase(ds.begin());
            saved = false;

            MLPL.resize(dl);
        }

    public:

        void (*activation)(float&);
        void (*activationDV)(float&);

        float lr = 0.1f;

        ANNA() : MLPL(0), tmp_layer(0), scale(0), ds(0), layers(0), dl(0), ddl(0), threads(1), saved(false), activation(MATH::none), activationDV(MATH::none) {};
        ANNA(std::vector<n> neurons, void (*_activation)(float&)) { this->init(neurons, _activation); }
        ANNA(std::vector<n> neurons, void (*_activation)(float&), void (*_activationDV)(float&)) { this->init(neurons, _activation, _activationDV); }

        void enableTraining(void (*_activationDV)(float&))
        {
            activationDV = _activationDV;
            tmp_layer.resize(dl);

            for (n l = 0; l < dl; ++l)
            {
                tmp_layer[l] = std::vector<float>(scale[l]);
                MLPL[l].create(ds[l], scale[l], true);
                MLPL[l].rand();
            }
        }

        void disableTraining()
        {
            activationDV = nullptr;
            tmp_layer.resize(0);

            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l], false);
                MLPL[l].rand();
            }
        }

        void init(std::vector<n> neurons, void (*_activation)(float&))
        {
            basicInit(neurons, _activation);
            disableTraining();
        }

        void init(std::vector<n> neurons, void (*_activation)(float&), void (*_activationDV)(float&))
        {
            basicInit(neurons, _activation);
            enableTraining(_activationDV);
        }

        n getNParameters()
        {
            n params = scale[0];
            for (n i = 1; i < dl; ++i) params += scale[i] + scale[i] * scale[i - 1];
            return params;
        }

        bool save(std::string path);
        void saveWarning(bool _save) { saved = !_save; }
        bool load(std::string path, void (*_activation)(float&));

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

        void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, n epochs);

        void operator=(const ANNA& other)
        {
            if (this == &other) return;

            this->MLPL = other.MLPL;
            this->tmp_layer = other.tmp_layer;
            this->scale = other.scale;
            this->ds = other.ds;
            this->layers = other.layers;
            this->dl = other.dl;
            this->ddl = other.ddl;
            this->threads = other.threads;
            this->activation = other.activation;
            this->activationDV = other.activationDV;
            this->saved = other.saved;
            this->lr = other.lr;
        }

        void destruct()
        {
            if (!saved)
            {
                std::cerr << "[ANNA-CPU ~ANNA()]: The current ANNA model hasn't been saved yet.\n"
                    "[ANNA-CPU ~ANNA()]: This warning can be turned of using `ANNA::saveWarning(false);`\n"
                    "[ANNA-CPU ~ANNA()]: Do you wish to save the model? (0: No, 1: Yes): ";
                bool choice = true;
                std::cin >> choice;

                if (choice)
                {
                    std::string filename = std::to_string(reinterpret_cast<std::uintptr_t>(this)) + std::string(".ANNA");
                    std::cout << "Trying to save under " << filename << "...\n";
                    if (this->save(filename)) std::cout << "Success.\n";
                    else std::cerr << "Failed.\n";
                }
                else std::cout << "[ANNA-CPU ~ANNA()]: Didn't save the model as per choice.\n";
                saveWarning(false);
            }

            MLPL.clear();
            tmp_layer.clear();
            scale.clear();
            ds.clear();

            layers = 0;
            dl = 0;
            ddl = 0;
            threads = 1;

            activation = nullptr;
            activationDV = nullptr;

            lr = 0.1;
        }

        ~ANNA() {}
    };
}

#endif