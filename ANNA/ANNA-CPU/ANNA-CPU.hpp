#ifndef ANNA_CPU_HPP
#define ANNA_CPU_HPP

#include <string>
#include <fstream>
#include <sstream>
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

        n layers, dl, ddl;
        n threads;

        void (*activation) (float&);
        void (*activationDV) (float&);

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
            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l]);
                MLPL[l].rand();
            }
        }

        void forward(const std::vector<float>& inp)
        {
            MLPL[0].forward(inp, activation, threads);
            for (n l = 1; l < dl; ++l)
                MLPL[l].forward(MLPL[l - 1].getState(), activation, threads);
        }

        void forwardForGrad(const std::vector<float>& inp)
        {
            tmp_layer[0] = inp;
            MLPL[0].forward(inp, activation, threads);
            for (n l = 1; l < dl; ++l)
                MLPL[l].forward(MLPL[l - 1].getState(), activation, threads);
        }

        void gradDesc(const std::vector<float>& expected_output)
        {
            tmp_layer[ddl] = MLPL[ddl].gradDesc(expected_output, MLPL[ddl - 1].getState(), activationDV, lr);
            for (n l = (ddl - 1); l > 0; --l)
                tmp_layer[l] = MLPL[l].gradDesc(tmp_layer[l + 1], MLPL[l - 1].getState(), activationDV, lr);
            MLPL[0].gradDesc(tmp_layer[1], tmp_layer[0], activationDV, lr);
        }

        void enableTraining(void (*_activationDV) (float&))
        {
            activationDV = _activationDV;
            tmp_layer.resize(dl);

            for (n l = 0; l < dl; ++l)
            {
                tmp_layer[l] = std::vector<float>(scale[l]);
                MLPL[l].enableTraining();
            }
        }

        void disableTraining()
        {
            activationDV = nullptr;
            tmp_layer.resize(0);

            for (auto& l : MLPL) l.disableTraining();
        }
    public:

        float lr = 0.1f;

        ANNA() : MLPL(0), tmp_layer(0), scale(0), ds(0), layers(0), dl(0), ddl(0), threads(1), activation(MATH::none), activationDV(MATH::none) {};
        ANNA(std::vector<n> neurons, void (*_activation) (float&)) { init(neurons, _activation); }

        std::vector<float> fw(std::vector<float>& inp, bool softmax)
        {
            forward(inp);
            if (softmax) calcSoftmaxOut();
            return getOutput();
        }

        void init(std::vector<n> neurons, void (*_activation) (float&))
        {
            basicInit(neurons, _activation);
            disableTraining();
        }
        
        void setThreads(n _threads)
        {
            threads = _threads;
            for (MLP::LAYER& l : MLPL) l.setThreads(threads);
        }

        n getNParameters()
        {
            n params = scale[0];
            for (n i = 1; i < dl; ++i) params += scale[i] + scale[i] * scale[i - 1];
            return params;
        }

        bool save(std::string path);
        bool load(std::string path, void (*_activation) (float&));

        const std::vector<float>& getOutput() { return MLPL[ddl].getState(); }
        void calcSoftmaxOut() { MLPL[ddl].softmax(); }

        void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, n epochs, void (*_activationDV) (float&));

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
            this->lr = other.lr;
        }

        ~ANNA()
        {
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

            lr = 0.1f;
        }
    };
}

#endif