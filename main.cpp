#include <iostream>
#include <chrono>
#include "HANNA/MLP/MLP.hpp"

using hrc = std::chrono::high_resolution_clock;
using timepoint = hrc::time_point;

timepoint tp;

void start() { tp = hrc::now(); }
void display()
{
    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(hrc::now() - tp).count() << "ms";
}

// Sigmoid
float activation(const float& x) { return 1.0f / (1.0f + std::exp(-x)); }
float activationdr(const float& x)
{
    float sigmoid = activation(x);
    return sigmoid * (1.0f - sigmoid);
}

int main()
{
    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    Eigen::initParallel();
    std::cout << "Eigen: THREADS_" << Eigen::nbThreads() << " & " << Eigen::SimdInstructionSetsInUse() << '\n';

    std::vector<std::vector<Eigen::VectorXf>> data(2);
    data[0] = std::vector<Eigen::VectorXf>(4, Eigen::VectorXf(2));
    data[1] = std::vector<Eigen::VectorXf>(4, Eigen::VectorXf(1));

    data[0][0] << 0.f, 0.f; data[0][1] << 0.f, 1.f;
    data[0][2] << 1.f, 0.f; data[0][3] << 1.f, 1.f;

    data[1][0] << 0.f; data[1][1] << 1.f;
    data[1][2] << 1.f; data[1][3] << 0.f;

    start();
    MLP::MLP mlp({ 2, 3, 1 });
    mlp.random();
    for (un e = 0; e < 100000; ++e)
    {
        for (un s = 0; s < 4; ++s)
        {
            mlp.forward(data[0][s], activation);
            mlp.graddesc(data[0][s], data[1][s], activationdr, 0.1f);
        }
    }
    display();
    std::cout << " to initialize & train " << mlp.get_param_count() << " parameters.\n";

    float error = 0.f;
    for (un s = 0; s < 4; ++s)
    {
        float value = mlp.report(data[0][s], activation)(0);
        error += (data[1][s](0) - value) / 4;
        std::cout << "(" << data[0][s](0) << " & " << data[0][s](1) << ") => " << value << '\n';
    }
    std::cout << "Error: " << error << std::endl;

    return 0;
}
