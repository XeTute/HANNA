#ifndef ANNA_CPU_HPP
#define ANNA_CPU_HPP

#include "MLP-Layer.hpp"
#include <iostream>

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

        void (*activation)(float&);
        void (*activationDV)(float&);

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

        float lr = 0.1f;

        ANNA() : MLPL(0), tmp_layer(0), scale(0), ds(0), layers(0), dl(0), ddl(0) {};
        ANNA(std::vector<n> neurons, void (*_activation)(float&))
        {
            basicInit(neurons, _activation);

            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l], false);
                std::cout << "Got to rand()\n";
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

        void forward(const std::vector<float>& inp, bool forGrad)
        {
            if (forGrad) tmp_layer[0] = inp;
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
    };
}

#endif