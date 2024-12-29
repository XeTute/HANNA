#include "ANNA-CPU.hpp"

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