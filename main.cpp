#include <iostream>
#include <string>

#include "HANNA/MLP/MLP.hpp"
#include "HANNA/activations.cpp"

void print(std::string _this) { std::cout << ">> " << _this << '\n'; }
void eraseable(std::string _this) { std::cout << ">> " << _this; }
void erase(std::string _this) { std::cout << "\r>> " << _this; }

std::string input(std::string str) // That's the only thing I like about Py
{
    std::cout << ">_ " + str;
    std::getline(std::cin, str); // str isn't a ref
    return str;
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

int main()
{
    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    Eigen::initParallel();

    // Config
    const std::vector<std::size_t> scale({ 128, 32, 4 });
    const unsigned short epochs = 100;
    const float learning_rate = 1e-2f;

    MLP::MLP net(scale);
    
    if (!net.load("emotionclassifier.mlp.bin"))
    {
        print("Failed to load model from disk: " + std::string(net.lastexception.what()));

        net.birth(scale);
        net.random();

        eraseable("Loading dataset...");
        std::vector<std::vector<std::string>> datastr = load_json("data.json");
        std::vector<std::vector<Eigen::VectorXf>> datanum(datastr.size(), std::vector<Eigen::VectorXf>(datastr[0].size()));
        erase("Loaded dataset.   \n");

        eraseable("Formatting dataset...");
        std::size_t rows = datastr.size();
        for (std::size_t row = 0; row < rows; ++row)
        {
            const std::string input  = datastr[row][0];
            const std::string output = datastr[row][1];

            Eigen::VectorXf numinp = str_to_ascii_vals(datastr[row][0], 128);
            datanum[row][0] = numinp;

            Eigen::VectorXf numout; numout.resize(4); numout.setZero();
                 if (output == "Sad"    ) numout(0) = 1.f;
            else if (output == "Happy"  ) numout(1) = 1.f;
            else if (output == "Neutral") numout(2) = 1.f;
            else if (output == "Angry"  ) numout(3) = 1.f;
            datanum[row][1] = numout;
        }
        erase("Formatted dataset.   \n");

        eraseable("Training model: Completed Epoch 0 / " + std::to_string(epochs) + ".");
        for (unsigned short epoch = 0; epoch < epochs; ++epoch)
        {
            for (std::size_t sample = 0; sample < rows; ++sample)
            {
                net.forward(datanum[sample][0], sigmoid);
                net.graddesc(datanum[sample][0], datanum[sample][1], sigmoid_derivative, learning_rate);
            }
            erase("Training model: Completed Epoch " + std::to_string(epoch) + " / " + std::to_string(epochs) + ".");
        }
        erase("Trained model.                              \n");

        eraseable("Saving model...");
        if (net.save("emotionclassifier.mlp.bin")) erase("Saved model successfully.");
        else erase("Failed to save model: " + std::string(net.lastexception.what()));
        std::cout << '\n';

        print("--- Absolute Eval ---");
        eraseable("Got 0 / " + std::to_string(rows) + " wrong.");
        std::size_t wrong = 0;
        for (std::size_t row = 0; row < rows; ++row)
        {
            Eigen::VectorXf out = net.report(datanum[row][0], sigmoid);
            unsigned short eoid = 0;

            for (unsigned short cid = 0; cid < 4; ++cid)
            {
                if (datanum[row][1](cid) == 1.f)
                {
                    eoid = cid;
                    break;
                }
            }

            if (out(eoid) < 0.25f)
            {
                ++wrong;
                erase("Got " + std::to_string(wrong) + " / " + std::to_string(rows) + " wrong.");
            }
        }
        std::cout << '\n';
    }

    print("--- Inference ---");
    while (true)
    {
        std::string prompt = input("");
        Eigen::VectorXf output = net.report(str_to_ascii_vals(prompt, 128), sigmoid);

        std::cout << ">> Classified as: ";
        if (output(0) >= 0.5f) std::cout << "Sad(" << output(0) << ") ";
        if (output(1) >= 0.5f) std::cout << "Happy(" << output(1) << ") ";
        if (output(2) >= 0.5f) std::cout << "Neutral(" << output(2) << ") ";
        if (output(3) >= 0.5f) std::cout << "Angry(" << output(3) << ") ";

        std::cout << "\n>> All distributions: "
                  << "Sad(" << output(0)
                  << "), Happy(" << output(1)
                  << "), Neutral(" << output(2)
                  << "), Angry(" << output(3) << ')';
        std::cout << "\n---\n";
    }

    return 0;
}
