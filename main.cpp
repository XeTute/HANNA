#include <iostream>
#include <chrono>

#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"
#include "ANNA/Data-Prep/CSV.hpp"
#include "ANNA/Data-Prep/normalize.hpp"

int main()
{
    auto data(CSV::loadCSVn<float>("HousingData.csv"));
    std::vector<float> data_flatten(0);

    n samples = data.size();
    n eps = data[0].size(); // elements per sample

    data_flatten.resize(samples * eps);
    for (n i = 0; i < samples; ++i)
    {
        for (n elem = 0; elem < eps; ++elem)
            data_flatten[i * eps + elem] = data[i][elem];
    }

    normalize::normConf<float> data_config(normalize::minMaxNorm(data_flatten));

    std::cout << "samples * eps = " << data_flatten.size() << ", data_config.x: " << data_config.x << ", data_config.y: " << data_config.y << '\n';
    std::cout << "Example sample:\n";
    for (n e = 0; e < eps; ++e)
        std::cout << "E" << e << ": " << data_flatten[e] << '\n';

    return 0;
}
