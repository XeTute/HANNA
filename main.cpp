#include <iostream>
#include <chrono>

#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"
#include "ANNA/Data-Prep/normalize.hpp"

using namespace std::chrono_literals;

const std::string dataset_path = "tiny-shakespeare.txt";
const n epochs = 1;
const n ctx_lngth = 128;

typedef std::vector<std::vector<float>> a2d;

void loadDataIntoString(std::string& o)
{
    std::ifstream r(dataset_path);
    std::string buf;

    r.seekg(0);
    o = "";

    while (std::getline(r, buf))
        o += buf + '\n';
}

int main()
{
    std::string datastr;
    loadDataIntoString(datastr);

    n samples = std::min(datastr.size(), size_t(std::pow(10, 6)));

    // Convert string seq. to trainable data for prediction
    std::vector<a2d> data(2, a2d(samples));
    for (n s = 0; s < samples; ++s)
    {
        n min = std::min(ctx_lngth, s);
        data[0][s] = std::vector<float>(ctx_lngth, 0.f);
        data[1][s] = std::vector<float>(255, 0.f);

        for (n c = 0; c < min; ++c) data[0][s][c] = (float)datastr[c + s];
        data[1][s][datastr[s + min]] = 10.f;
    }

    normalize::normConf<float> conf;
    {
        std::vector<float> seq(255, 0.f);
        for (n i = 0; i < 255; ++i) seq[i] = float(i);
        conf = normalize::minMaxNorm(seq);
    }
    for (std::vector<float>& sample : data[0])
        normalize::minMaxNorm(sample, conf);

    ANNA_CPU::ANNA model({ ctx_lngth, ctx_lngth / 2, ctx_lngth / 3, 255 }, MATH::sigmoid, MATH::sigmoidDv);
    model.lr = 0.01;
    model.setThreads(6);
    std::cout << model.getNParameters() << " Parameters will be trained on " << samples << '.';
    
    std::chrono::high_resolution_clock::time_point tp[2] = { std::chrono::high_resolution_clock::now() };
    model.train(data[0], data[1], epochs);
    tp[1] = std::chrono::high_resolution_clock::now();
    std::cout << "\nTook " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms.";

    while (true) // Inference, assuming the input is under ctx_lngth - 8 characters long
    {
        std::string inpstr;
        std::vector<float> inp(ctx_lngth, 0.f);
        
        std::cout << "\n> ";
        std::getline(std::cin, inpstr);
        for (n c = 0; c < inpstr.size(); ++c) inp[c] = (float)inpstr[c];

        for (n gen = 0; gen < 8; ++gen)
        {
            model.forward(inp);

            float out = 0.f;
            float oldmax = 0.f;
            for (n i = 0; i < 255; ++i)
            {
                if (oldmax < model.getOutput()[i])
                {
                    oldmax = model.getOutput()[i];
                    out = i;
                }
            }

            std::cout << (char)out;
            inp[inpstr.size() + gen] = out;
        }
    }

    return 0;
}