#include <iostream>
#include <chrono>

#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"
#include "ANNA/Data-Prep/CSV.hpp"
#include "ANNA/Data-Prep/normalize.hpp"

int main()
{
    std::vector<std::vector<float>> data(CSV::loadCSVn<float>("HousingDataNormalized.csv"));
    n samples = 0;
    n eps = 0; // elems per sample
    std::vector<std::vector<float>> input;
    std::vector<std::vector<float>> output;

    // for (std::vector<float>& sample : data)
    //     normalize::minMaxNorm(sample, {0, 711});
    // CSV::saveCSVn<float>(data, "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV", "HousingDataNormalized.csv");

    samples = data.size();
    eps = data[0].size();

    input.resize(samples, std::vector<float>(eps - 1, 0.f));
    output.resize(samples, std::vector<float>(1, 0.f));

    for (n i = 0; i < samples; ++i)
    {
        output[i] = std::vector<float>(1, data[i][9]); // data[i][9] = TAX
        data[i].erase(data[i].begin() + 9);
        input[i] = data[i];
    }

    ANNA_CPU::ANNA model({ eps - 1, (eps - 1) / 2, 1 }, MATH::sigmoid, MATH::sigmoidDv);
    model.lr = 0.075;
    model.setThreads(6);
    model.train(input, output, 100);

    if (!model.save("HousingData.ANNA"))
    {
        std::cout << "Failed to save the model. Turning off warning...\n";
        model.saveWarning(false);
        std::cout << "Done.\n";
    }

    return 0;
}