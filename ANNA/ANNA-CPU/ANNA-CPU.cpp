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
    saveWarning(false);
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
    
    std::vector<n> mlp_scale(scale);
    mlp_scale.erase(mlp_scale.begin());
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
        MLPL[l].pretrained(bias, weight_ll, true);
    }
    return true;
}

bool ANNA_CPU::ANNA::load(std::string path, void (*_activation)(float&), void (*_activationDV)(float&))
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
    enableTraining(_activationDV);

    std::vector<n> mlp_scale(scale);
    mlp_scale.erase(mlp_scale.begin());
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
        MLPL[l].pretrained(bias, weight_ll, true);
    }
    return true;
}

void ANNA_CPU::ANNA::train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, n epochs)
{
    if (input.size() != output.size())
    {
        std::cerr << "[ANNA-CPU train(...)]: The sizes of input & output do not match. Won't train.\n";
        return;
    }

    n samples = input.size();
    for (n i = 0; i < samples; ++i)
    {
        if (input[i].size() != scale[0])
        {
            std::cout << "[ANNA-CPU train(...)]: The size of input[" << i << "] doesn't match with the network's input size. Won't train.\n";
            return;
        }
        if (output[i].size() != scale[dl])
        {
            std::cout << "[ANNA-CPU train(...)]: The size of output[" << i << "] doesn't match with the network's output size. Won't train.\n";
            return;
        }
    }

    if (threads == 1)
    {
        for (n e = 0; e < epochs; ++e)
        {
            for (n s = 0; s < samples; ++s)
            {
                this->forwardForGrad(input[s]);
                this->gradDesc(output[s]);
            }
        }
        return;
    }

    n chunkSize = std::ceil(samples / threads);
    n remainder = samples % threads;

    std::vector<std::thread> pool(threads);
    std::vector<ANNA> copy(threads, *this);

    for (n thread = 0; thread < threads; ++thread)
    {
        pool[thread] = std::thread
        (
            [&](n id)
            {
                n end = id * chunkSize + chunkSize;
                if ((threads - 1) == id) end += remainder;
                for (n e = 0; e < epochs; ++e)
                {
                    for (n s = id * chunkSize; s < end; ++s)
                    {
                        copy[id].forwardForGrad(input[s]);
                        copy[id].gradDesc(output[s]);
                    }
                }
            }, thread
        );
    }

    n iterations = (threads >= dl) ? dl : threads;

    for (std::thread& thread : pool)
        if (thread.joinable()) thread.join();

    for (n l = 0; l < iterations; ++l)
    {
        pool[l] = std::thread
        (
            [&, l]
            {
                n neurons = MLPL[l].getBias().size();
                n nll = MLPL[l].getWeights()[0].size();
                float sum = 0.f;

                for (n neuron = 0; neuron < neurons; ++neuron)
                {
                    sum = 0.f;
                    for (ANNA& c : copy)
                        sum += c.MLPL[l].getBias()[neuron];
                    MLPL[l].getBias()[neuron] = sum / threads;

                    for (n _nll = 0; _nll < nll; ++_nll)
                    {
                        sum = 0.f;
                        for (ANNA& c : copy)
                            sum += c.MLPL[l].getWeights()[neuron][_nll];
                        MLPL[l].getWeights()[neuron][_nll] = sum / threads;
                    }
                }
            }
        );
    }

    for (n l = iterations; l < dl; l++)
    {
        n neurons = MLPL[l].getBias().size();
        n nll = MLPL[l].getWeights()[0].size();
        float sum = 0.f;

        for (n neuron = 0; neuron < neurons; ++neuron)
        {
            sum = 0.f;
            for (ANNA& c : copy)
                sum += c.MLPL[l].getBias()[neuron];
            MLPL[l].getBias()[neuron] = sum / threads;

            for (n _nll = 0; _nll < nll; ++_nll)
            {
                sum = 0.f;
                for (ANNA& c : copy)
                    sum += c.MLPL[l].getWeights()[neuron][_nll];
                MLPL[l].getWeights()[neuron][_nll] = sum / threads;
            }
        }
    }

    for (std::thread& thread : pool)
        if (thread.joinable()) thread.join();
}