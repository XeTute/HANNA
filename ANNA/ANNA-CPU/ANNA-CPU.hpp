#ifndef ANNA_CPU_HPP
#define ANNA_CPU_HPP

#include "MLP-Layer.hpp"
#include <string>
#include <fstream>
#include <sstream>

namespace ANNA_CPU
{
    class ANNA
    {
    private:

        std::vector<MLP::LAYER> MLPL;
        std::vector<std::vector<float>> tmp_layer;
        std::vector<n> scale;
        std::vector<n> ds; // decreased scale, scale without the first element for more effective hidden & output layer construction

        n layers;
        n dl; // decreased layers
        n ddl;

        unsigned short threads;

        void basicInit(std::vector<n> neurons, void (*_activation) (float&))
        {
            scale = neurons;

            layers = neurons.size();
            dl = layers - 1;
            ddl = layers - 2;
            activation = _activation;

            ds = neurons;
            ds.erase(ds.begin());
            saved = false;

            MLPL.resize(dl);
        }

        bool saved = false;
    public:

        void (*activation)(float&);
        void (*activationDV)(float&);

        float lr = 0.1f;

        ANNA() : MLPL(0), tmp_layer(0), scale(0), ds(0), layers(0), dl(0), ddl(0), threads(1), saved(false) {};
        ANNA(std::vector<n> neurons, void (*_activation)(float&))
        {
            basicInit(neurons, _activation);

            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l], false);
                MLPL[l].rand();
            }
        }

        ANNA(std::vector<n> neurons, void (*_activation)(float&), void (*_activationDV)(float&))
        {
            basicInit(neurons, _activation);

            activationDV = _activationDV;
            tmp_layer.resize(dl);

            for (n l = 0; l < dl; ++l)
            {
                MLPL[l].create(ds[l], scale[l], true);
                MLPL[l].rand();
            }
        }

        bool save(std::string path)
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
            saved = true;

            w.close();
            return true;
        }
        void saveWarning(bool _save) { saved = _save; }

        bool load(std::string path, void (*_activation)(float&))
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

        void display()
        {
            std::cout << "\n\nDISPLAY START";
            for (n l = 0; l < dl; ++l)
            {
                std::cout << "\n<--- LAYER " << l << " --->\n";
                std::cout << "Neuron-Count: " << MLPL[l].getState().size() << '\n';
                std::cout << "Bias: \n";
                for (float bias : MLPL[l].getBias()) std::cout << "- " << bias << '\n';

                std::cout << "\nWeights: \n";
                for (n neuron = 0; neuron < MLPL[l].getBias().size(); ++neuron)
                {
                    std::cout << ">-- NEURON " << neuron << " <--\n";
                    for (n nll = 0; nll < MLPL[l].getWeights()[0].size(); ++nll)
                        std::cout << "- " << MLPL[l].getWeights()[neuron][nll] << '\n';
                }
            }
            std::cout << "DISPLAY END\n\n";
        }

        void setThreads(unsigned short _threads) { threads = _threads; }
        unsigned short getThreads() { return threads; }

        void forward(const std::vector<float>& inp)
        {
            MLPL[0].forward(inp, activation);
            for (n l = 1; l < dl; ++l)
                MLPL[l].forward(MLPL[l - 1].getState(), activation);
        }

        void forwardForGrad(const std::vector<float>& inp)
        {
            tmp_layer[0] = inp;
            MLPL[0].forward(inp, activation);
            for (n l = 1; l < dl; ++l)
                MLPL[l].forward(MLPL[l - 1].getState(), activation);
        }

        void gradDesc(const std::vector<float>& expected_output)
        {
            tmp_layer[ddl] = MLPL[ddl].gradDesc(expected_output, MLPL[ddl - 1].getState(), activationDV, lr);
            for (n l = (ddl - 1); l > 0; --l)
                tmp_layer[l] = MLPL[l].gradDesc(tmp_layer[l + 1], MLPL[l - 1].getState(), activationDV, lr);
            MLPL[0].gradDesc(tmp_layer[1], tmp_layer[0], activationDV, lr);
        }

        const std::vector<float>& getOutput() { return MLPL[ddl].getState(); }
        void calcSoftmaxOut() { MLPL[ddl].softmax(); }

        void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, n EPOCHS)
        {
            if (input.size() != output.size())
            {
                std::cerr << "[ANNA-CPU train(...)]: The sizes of input & output do not match. Won't train.\n";
                return;
            }

            n samples = input.size();

            if (threads == 1)
            {
                for (n e = 0; e < EPOCHS; ++e)
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

            std::vector<ANNA> copy(threads, *this);
        }

        void operator=(const ANNA& other)
        {
            if (this == &other) return;

            this->MLPL = other.MLPL;
            this->tmp_layer = other.tmp_layer;
            this->scale = other.scale;
            this->ds = other.ds;
            this->layers = other.layers;
            this->dl = other.dl;
            this->ddl = other.ddl;
            this->threads = other.threads;
            this->activation = other.activation;
            this->activationDV = other.activationDV;
            this->saved = other.saved;
            this->lr = other.lr;
        }

        ~ANNA()
        {
            if (!saved)
            {
                std::cerr << "[ANNA-CPU ~ANNA()]: The current ANNA model hasn't been saved yet.\n"
                             "[ANNA-CPU ~ANNA()]: This warning can be turned of using `ANNA::saveWarning(false);`\n"
                             "[ANNA-CPU ~ANNA()]: Do you wish to save the model? (0: No, 1: Yes): ";
                bool choice = true;
                std::cin >> choice;

                if (choice)
                {
                    std::string filename = std::to_string(reinterpret_cast<std::uintptr_t>(this)) + std::string(".ANNA");
                    std::cout << "Trying to save under " << filename << "...\n";
                    if (this->save(filename)) std::cout << "Success.\n";
                    else std::cerr << "Failed.\n";
                }
                else std::cout << "[ANNA-CPU ~ANNA()]: Didn't save the model as per choice.\n";
            }

            MLPL.clear();
            tmp_layer.clear();
            scale.clear();
            ds.clear();

            layers = 0;
            dl = 0;
            ddl = 0;
            threads = 1;

            activation = nullptr;
            activationDV = nullptr;
            
            lr = 0.1;
        }
    };
}

#endif
