#include <iostream>
#include <chrono>

#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"

using namespace ANNA_CPU;

constexpr n EPOCHS = 10000;

int main()
{
    ANNA model({2, 3, 1}, MATH::sigmoid, MATH::sigmoidDv);
    
    bool load = false;
    std::cout << "> Load?: ";
    std::cin >> load;

    std::vector<std::vector<std::vector<float>>> data
    {
        { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
        { { 0 }, { 1 }, { 1 }, { 0 } }
    };
    n SAMPLES = data[0].size();

    if (load)
    {
        if (!model.load("model.ANNA", MATH::sigmoid)) std::cerr << "Loading model failed.\n";
        else std::cout << "Loaded model.\n";
    }
    
    else
    {
        model.lr = 0.1f;

        std::cout << "Starting training...\n";
        std::chrono::time_point<std::chrono::high_resolution_clock> tp[2] = { std::chrono::high_resolution_clock::now() };
        for (n epoch = 0; epoch < EPOCHS; ++epoch)
        {
            for (n sample = 0; sample < SAMPLES; ++sample)
            {
                model.forwardForGrad(data[0][sample]);
                model.gradDesc(data[1][sample]);
            }
        }
        tp[1] = std::chrono::high_resolution_clock::now();

        std::cout << "Training finished.\nTook " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms\n>--- RESULTS ---<\n";
    }

    for (n sample = 0; sample < SAMPLES; ++sample)
    {
        std::cout << "I: " << data[0][sample][0] << " : " << data[0][sample][1] << " => ";
        model.forward(data[0][sample]);
        auto o = model.getOutput();
        for (float& x : o) std::cout << x;
        std::cout << '\n'; 
    }

    return 0;
}
