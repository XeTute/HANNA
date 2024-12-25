#include <iostream>
#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"

using namespace ANNA_CPU;

constexpr n EPOCHS = 10000;

int main()
{
    ANNA model({2, 2, 1}, MATH::sigmoid, MATH::sigmoidDv);
    model.lr = 0.1f;

    std::vector<std::vector<std::vector<float>>> data
    {
        { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
        { { 0 }, { 1 }, { 1 }, { 0 } }
    };
    n SAMPLES = data[0].size();

    std::cout << "Starting training...\n";
    for (n epoch = 0; epoch < EPOCHS; ++epoch)
    {
        for (n sample = 0; sample < SAMPLES; ++sample)
        {
            model.forward(data[0][sample], true);
            model.gradDesc(data[1][sample]);
        }
    }

    std::cout << "Training finished.\n>--- RESULTS ---<\n";
    for (n sample = 0; sample < SAMPLES; ++sample)
    {
        std::cout << "I: " << data[0][sample][0] << " : " << data[0][sample][1] << " => ";
        model.forward(data[0][sample], false);
        auto o = model.getOutput();
        for (float& x : o) std::cout << x;
        std::cout << '\n'; 
    }

    return 0;
}