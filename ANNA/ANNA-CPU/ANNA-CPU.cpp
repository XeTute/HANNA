#include "ANNA-CPU.hpp"

bool ANNA_CPU::ANNA::save(std::string path)
{
    std::ofstream w(path, std::ios::out | std::ios::binary);
    if (!w.is_open() || !w) return false;
    for (n neurons : scale) w << neurons << ';';
    w << '\n';
    for (n l = 0; l < dl; ++l)
    {
        if (!w.write(reinterpret_cast<const char*>(MLPL[l].getBias().data()), MLPL[l].getBias().size() * sizeof(float)))
        {
            std::cerr << "Failed to read: std::ofstream returned false while trying to write binary. " << path << " may be corrupted.\n";
            return false;
        }
        for (n neuron = 0; neuron < MLPL[l].getWeights().size(); ++neuron)
        {
            if (!w.write(reinterpret_cast<const char*>(MLPL[l].getWeights()[neuron].data()), MLPL[l].getWeights()[neuron].size() * sizeof(float)))
            {
                std::cerr << "Failed to read: std::ofstream returned false while trying to write binary. " << path << " may be corrupted.\n";
                return false;
            }
        }
    }
    w.close();
    return true;
}

bool ANNA_CPU::ANNA::load(std::string path, void (*_activation)(float&))
{
    std::ifstream r(path, std::ios::in | std::ios::binary);
    if (!r.is_open() || !r) return false;
    r.seekg(0);
    scale = std::vector<n>(0);
    {
        std::string strBuffer("");
        std::stringstream ssBuffer("");
        char charBuffer = 0;
        std::getline(r, strBuffer);
        ssBuffer = std::stringstream(strBuffer);
        
        while (std::getline(ssBuffer, strBuffer, ';'))
            scale.push_back(std::stoull(strBuffer));
    }

    basicInit(scale, activation);
    std::vector<n> mlp_scale(ds);

    for (n l = 0; l < dl; ++l)
    {
        std::vector<float> bias(mlp_scale[l], 0.f);
        std::vector<std::vector<float>> weight_ll(mlp_scale[l], std::vector<float>(scale[l], 1.f));
        if (!r.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float)))
        {
            std::cerr << "Failed to read: std::ifstream returned false while trying to read binary. ANNA may be corrupted.\n";
            return 0;
        }
        for (std::vector<float>& neuron : weight_ll)
        {
            if (!r.read(reinterpret_cast<char*>(neuron.data()), neuron.size() * sizeof(float)))
            {
                std::cerr << "Failed to read: std::ifstream returned false while trying to read binary. ANNA may be corrupted.\n";
                return 0;
            }
        }
        MLPL[l].pretrained(bias, weight_ll);
    }
    return true;
}

void ANNA_CPU::ANNA::train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, n epochs, void (*_activatioDV) (float&))
{
    n samples = input.size();
    if (samples != output.size())
    {
        std::cerr << "[ANNA-CPU train(...)]: The sizes of input & output do not match. Won't train.\n";
        return;
    }

    enableTraining(_activatioDV);
    for (n e = 0; e < epochs; ++e)
    {
        for (n s = 0; s < samples; ++s)
        {
            forwardForGrad(input[s]);
            gradDesc(output[s]);
        }
    }
    disableTraining();
}