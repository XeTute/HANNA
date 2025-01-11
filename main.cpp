#include <iostream>
#include <chrono>

#include "ANNA/ANNA-CPU/ANNA-CPU.hpp"
#include "ANNA/Data-Prep/normalize.hpp"

using namespace std::chrono_literals;

const std::string dataset_path = "tiny-shakespeare.txt";
const n epochs = 2;
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

    n samples = datastr.size();

    // Convert string seq. to trainable data for prediction
    samples -= ctx_lngth;
    std::vector<a2d> data(2, a2d(samples));
    for (n s = 0; s < samples; ++s)
    {
        n min = std::min(ctx_lngth, s);
        data[0][s] = std::vector<float>(ctx_lngth, 0.f);
        data[1][s] = std::vector<float>(255, 0.f);

        for (n c = 0; c < min; ++c)
            data[0][s][c] = (float)datastr[c + s];
        data[1][s][datastr[s + min]] = 1.f;
    }
    samples += ctx_lngth;

    normalize::normConf<float> conf;
    {
        std::vector<float> seq(255, 0.f);
        for (n i = 0; i < 255; ++i) seq[i] = float(i);
        conf = normalize::minMaxNorm(seq);
    }
    for (std::vector<float>& sample : data[0])
        normalize::minMaxNorm(sample, conf);

    ANNA_CPU::ANNA model({ ctx_lngth, ctx_lngth / 2, ctx_lngth / 4, 255 }, MATH::ReLU);
    model.lr = 0.01f;
    model.setThreads(6);

    std::cout << model.getNParameters() << " Parameters will be trained on " << samples << " Samples.\n";

    std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
    model.train(data[0], data[1], epochs, MATH::ReLU);

    std::cout << "\nTook " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tp).count() << "ms.";

    while (true) // Inference, assuming the input is under ctx_lngth - 8 characters long
    {
        std::string inpstr;
        std::vector<float> inp(ctx_lngth, 0.f);
        
        std::cout << "\n> ";
        std::getline(std::cin, inpstr);
        for (n c = 0; c < inpstr.size(); ++c) inp[c] = (float)inpstr[c];

        std::vector<float> out = model.fw(inp, false);
        for (n gen = 0; gen < 8; ++gen)
        {
            n max = 0;
            float oldmax = 0.f;
            for (n i = 0; i < 255; ++i)
            {
                if (oldmax < out[i])
                {
                    oldmax = out[i];
                    max = i;
                }
            }

            std::cout << (char)max;
            inp[inpstr.size() + gen] = float(max);
        }
    }

    return 0;
}