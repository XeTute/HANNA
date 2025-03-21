#include <iostream>
#include <chrono>
#include "HANNA/MLP/MLP.hpp"

using hrc = std::chrono::high_resolution_clock;
using timepoint = hrc::time_point;

timepoint tp;

void start() { tp = hrc::now(); }
void display()
{
    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(hrc::now() - tp).count() << "ms.\n";
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

    MLP::MLP mlp({ 2, 2, 1 });
    mlp.random();
    std::cout << "Parameter Count: " << mlp.get_param_count() << '\n';

    std::vector<std::vector<Eigen::VectorXf>> data(2);
    data[0] = std::vector<Eigen::VectorXf>(4, Eigen::VectorXf(2));
    data[1] = std::vector<Eigen::VectorXf>(4, Eigen::VectorXf(1));

    data[0][0] << 0.f, 0.f; data[0][1] << 0.f, 1.f;
    data[0][2] << 1.f, 0.f; data[0][3] << 1.f, 1.f;

    data[1][0] << 0.f; data[1][1] << 1.f;
    data[1][2] << 1.f; data[1][3] << 0.f;

    start();
    if (!mlp.load("XOR.mlp"))
    {
		std::cout << "Failed loading: " << mlp.lastexception.what() << '\n';

		// Failed loading will invalidate model parameters
		mlp.birth({ 2, 2, 1 });
		mlp.random();
        for (un e = 0; e < 100000; ++e)
        {
            for (un s = 0; s < 4; ++s)
            {
                mlp.forward(data[0][s], activation);
                mlp.graddesc(data[0][s], data[1][s], activationdr, 0.1f);
            }
        }
		std::cout << "Following measures are for how long it took to train the model.\n";
    }
	else std::cout << "Following measures are for how long it took to load the model.\n";
    display();

    float error = 0.f;
    for (un s = 0; s < 4; ++s)
    {
        float value = mlp.report(data[0][s], activation)(0);
        error += data[1][s](0) - value;
        std::cout << "(" << data[0][s](0) << " & " << data[0][s](1) << ") => " << value << '\n';
    }
    std::cout << "Error: " << error << std::endl;
	mlp.save("XOR.mlp");

    return 0;
}