#include <iostream>
#include <chrono>

#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"
#include "ANNA/Data-Prep/CSV.hpp"
#include "ANNA/Data-Prep/normalize.hpp"

const std::string dataset_path = "HSC-Small.csv"; // Hate-Speech-Classification
const n epochs = std::pow(10, 4);

int main()
{
    auto data(CSV::loadCSVstr(dataset_path)); // std::vector<std::vector<std::string>>
    n samples = data.size();

    std::vector<std::vector<float>> input(samples, std::vector<float>(0));
    std::vector<std::vector<float>> output(samples, std::vector<float>(1, 0.f));

    for (n s = 0; s < samples; ++s)
    {
        n chars = data[s][0].size();
        input[s].resize(chars);
        for (n c = 0; c < chars; ++c)
            input[s][c] = (unsigned int)data[s][0][c]; // To simply get ASCII values, C-Style casts are save?

        output[s][0] = std::stof(data[s][1]);
    }
    std::cout << "Initialized " << samples << " samples.\n";

    ANNA_CPU::ANNA model(std::vector<n>({ 48, 36, 24, 13, 1 }), MATH::sigmoid, MATH::sigmoidDv);
    model.setThreads(6);

    std::cout << "Starting training on " << samples << " samples & on " << epochs << " epochs...\n";
    model.train(input, output, epochs);
    std::cout << "Training finished...\n";

    if (!model.save(dataset_path.substr(0, dataset_path.size() - 3) + "ANNA"))
        std::cout << "Failed to save the model =(";
    else
        std::cout << "Saved the ANNA model successfully =)";
    std::cout << std::endl;

    return 0;
}