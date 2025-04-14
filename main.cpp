#include <iostream>
#include <string>

#include "HANNA/MLP/MLP.hpp"
#include "HANNA/mathfunctions.cpp"

void print(std::string _this) { std::cout << ">> " << _this << '\n'; }
void eraseable(std::string _this) { std::cout << ">> " << _this; }
void erase(std::string _this) { std::cout << "\r>> " << _this; }

std::string input(std::string str) // That's the only thing I like about Py
{
    std::cout << ">_ " + str;
    std::getline(std::cin, str); // str isn't a ref
    return str;
}

std::string remove_instructions(std::string path)
{
    std::string cleaned(path.substr(0, path.find(".json")) + "_noinstruction.json");
    std::string buffer("");
    std::ifstream r(path);
    std::ofstream w(cleaned);

    while (std::getline(r, buffer))
    {
        if (buffer.find("\"instruction\": ") == std::string::npos)
            w << buffer << '\n';
    }

    return cleaned;
}

std::vector<std::vector<std::string>> load_json(std::string path) // 10 min work, naive but works for this dataset
{
    std::vector<std::vector<std::string>> data(0);
    std::ifstream r(path);
    std::string line("");

    if (!r.good()) std::cout << "\rFailed to load data from " << path << ", expect a crash.";
    while (std::getline(r, line))
    {
        const std::size_t starts[3] = { line.find("\"input\": \""), line.find("\"output\": \"") };

        if (starts[0] != std::string::npos)
        {
            line = line.substr(starts[0] + std::string("\"input\": \"").size());
            line = line.substr(0, line.size() - 2);
            data.push_back(std::vector<std::string>({ line }));
        }
        else if (starts[1] != std::string::npos)
        {
            line = line.substr(starts[1] + std::string("\"output\": \"").size());
            line = line.substr(0, line.size() - 1);
            data[data.size() - 1].push_back(line);
        }
    }
    return data;
}

Eigen::VectorXf str_to_ascii_vals(const std::string& str, std::size_t vecsize)
{
    Eigen::VectorXf vec; vec.resize(vecsize); vec.setZero();
    std::size_t strsize = std::min(str.size(), vecsize);

    for (std::size_t x = 0; x < strsize; ++x)
        vec(x) = float(int(str[x])) / 128.f;
    return vec;
}

unsigned short get_pred_nrn_num(const Eigen::VectorXf& vec)
{
    unsigned short id = 0;
    float value = 0.f;
    unsigned short size(vec.size());

    for (unsigned short x = 0; x < size; ++x)
    {
        if (vec(x) > value)
        {
            id = x;
            value = vec(x);
        }
    }

    return id;
}

int main()
{
    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    Eigen::initParallel();

    // Config
    const std::size_t context_length = 128;
    const std::vector<std::size_t> scale({ context_length, 16, 64, 4 });
    unsigned short epochs = 10;
    float learning_rate = 1e-1f;

    MLP::MLP net;
    if (!net.load("emotionclassifier.mlp.bin"))
    {
        net.birth(scale);
        net.random();

        print("Failed to load model from disk: " + std::string(net.lastexception.what()));
        print("Will train a MLP with " + std::to_string(net.get_param_count()) + " params.");

        eraseable("Loading dataset...");
        std::vector<std::vector<std::string>> datastr = load_json("data.json");
        std::vector<std::vector<Eigen::VectorXf>> datanum(datastr.size(), std::vector<Eigen::VectorXf>(datastr[0].size()));
        erase("Loaded dataset with " + std::to_string(datastr.size()) + " rows.\n");

        eraseable("Formatting dataset...");
        std::size_t rows = datastr.size();
        for (std::size_t row = 0; row < rows; ++row)
        {
            const std::string input  = datastr[row][0];
            const std::string output = datastr[row][1];

            Eigen::VectorXf numinp = str_to_ascii_vals(datastr[row][0], context_length);
            datanum[row][0] = numinp;

            Eigen::VectorXf numout; numout.resize(4); numout.setZero();
                 if (output == "Sad"    ) numout(0) = 1.f;
            else if (output == "Happy"  ) numout(1) = 1.f;
            else if (output == "Neutral") numout(2) = 1.f;
            else if (output == "Angry"  ) numout(3) = 1.f;
            datanum[row][1] = numout;
        }
        erase("Formatted dataset.   \n");

        // Train
        eraseable("Training model: Completed Epoch 0 / " + std::to_string(epochs) + ".");

        float locallr = learning_rate;
        for (unsigned short epoch = 0; epoch < epochs; ++epoch)
        {
            for (std::size_t sample = 0; sample < rows; ++sample)
            {
                net.forward(datanum[sample][0], sigmoid);
                net.graddesc(datanum[sample][0], datanum[sample][1], sigmoid_derivative, locallr);
            }
            locallr *= 0.99f;
            erase("Training model: Completed Epoch " + std::to_string(epoch) + " / " + std::to_string(epochs) + " : lr set to " + std::to_string(locallr));
        }
        erase("Trained model.                                             \n"); // 50 spaces

        // Eval
        print("--- Absolute Eval ---");
        eraseable("Got 0 / " + std::to_string(rows) + " wrong.");

        std::size_t wrong = 0;
        for (std::size_t row = 0; row < rows; ++row)
        {
            Eigen::VectorXf out = net.report(datanum[row][0], sigmoid);

            if (datanum[row][1][get_pred_nrn_num(out)] != 1.0f)
            {
                ++wrong;
                erase("Got " + std::to_string(wrong) + " / " + std::to_string(rows) + " wrong.");
            }
        }
        std::cout << '\n';

        eraseable("Saving model...");
        if (net.save("emotionclassifier.mlp.bin")) erase("Saved model successfully.");
        else erase("Failed to save model: " + std::string(net.lastexception.what()));
        std::cout << '\n';
    }

    print("--- Inference ---");
    while (true)
    {
        std::string prompt = input("");
        Eigen::VectorXf output = net.report(str_to_ascii_vals(prompt, context_length), sigmoid);
        unsigned short strongest_neuron = get_pred_nrn_num(softmax(output));

        std::cout << ">> Classified as: ";
             if (strongest_neuron == 0) std::cout << "Sad(" << output(0) << ") ";
        else if (strongest_neuron == 1) std::cout << "Happy(" << output(1) << ") ";
        else if (strongest_neuron == 2) std::cout << "Neutral(" << output(2) << ") ";
        else if (strongest_neuron == 3) std::cout << "Angry(" << output(3) << ") ";

        std::cout << "\n>> All distributions: "
                  << "Sad(" << output(0)
                  << "), Happy(" << output(1)
                  << "), Neutral(" << output(2)
                  << "), Angry(" << output(3) << ')';
        std::cout << "\n---\n";
    }

    return 0;
}

/*int main()
{
    std::cout << "New filename: " << remove_instructions("data.json") << std::endl;
    return 0;
}*/