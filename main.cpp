#include <iostream>
#include "ANNA/ANNA-CPU/MLP-Layer.hpp"

void activation(float& x)
{
    if ((x >= 1.f) || (x <= -1.f)) x *= 0.7;
}

int main()
{
    const n layers = 5;
    const n dLayers = layers - 1;
    n scale[layers] = { 2, 8, 8, 4, 2 };
    n scale_mlp[dLayers] = { 8, 8, 4, 2 };

    MLP::LAYER MLPL[dLayers];

    for (n i = 0; i < dLayers; ++i)
    {
        MLPL[i].create(scale_mlp[i], scale[i]);
        MLPL[i].setRand();
    }

    float x[2] = { 0.f, 0.f };
    std::cout << "<< ";
    std::cin >> x[0] >> x[1];
    

    MLPL[0].forward({ 1.f, 0.f }, activation);
    for (n i = 1; i < dLayers; ++i)
        MLPL[i].forward(MLPL[i - 1].getState(), activation);
    
    MLPL[dLayers - 1].softmax();
    for (float y : MLPL[dLayers - 1].getState())
        std::cout << y << '\n';

    return 0;
}