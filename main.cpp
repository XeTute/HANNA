#include <iostream>
#include "HANNA/MLP/MLP.hpp"

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
    std::cout << "Eigen: THREADS_" << Eigen::nbThreads() << " & " << Eigen::SimdInstructionSetsInUse() << '\n';

    MLP::MLP mlp({ 2, 20, 20, 1 });
    mlp.random();

    std::vector<std::vector<Eigen::VectorXf>> data =
    {
        { Eigen::VectorXf(2), Eigen::VectorXf(2), Eigen::VectorXf(2), Eigen::VectorXf(2) },
        { Eigen::VectorXf(1), Eigen::VectorXf(1), Eigen::VectorXf(1), Eigen::VectorXf(1) }
    };

    data[0][0] << 0.f, 0.f;
    data[0][1] << 0.f, 1.f;
    data[0][2] << 1.f, 0.f;
    data[0][3] << 1.f, 1.f;

    data[1][0] << 0.f;
    data[1][1] << 1.f;
    data[1][2] << 1.f;
    data[1][3] << 0.f;

    for (un e = 0; e < 1000000; ++e)
    {
        for (un s = 0; s < 4; ++s)
        {
            mlp.forward(data[0][s], activation);
            mlp.graddesc(data[0][s], data[1][s], activationdr, 0.1f);
        }
    }

    for (un s = 0; s < 4; ++s)
        std::cout << "(" << data[0][s](0) << " & " << data[0][s](1) << ") => " << mlp.report(data[0][s], activation)(0) << '\n';

    return 0;
}