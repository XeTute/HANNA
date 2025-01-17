#include <iostream>
#include <chrono>
#include <cmath>
#include "HANNA/HANNA.hpp"

// Test 0.01 for HANNA
// Test trains a MLP model on predicting / generating shakespeare-like text, after training it
// Not Suitable for production usages. Consider using small Transformer, RNN or CNN models for that.
// Test-Code licensed under MIT, HANNA Library under Apache2.0.
// Port to OpenCL expected in four - x months.

void sigmoid(float& x) { x = 1.f / (1.f + std::exp(-x)); }
void sigmoidDV(float& x) { x *= (1 - x); }

constexpr std::size_t ctx = 12;

int main()
{
	std::chrono::high_resolution_clock::time_point tp[2]; // Benchmarking

	// Parse data from file
	tp[0] = std::chrono::high_resolution_clock::now();
	std::string strdata("");
	{
		std::ifstream r("tiny-shakespeare.txt", std::ios::in);
		std::string buf;
		while (std::getline(r, buf))
			strdata += buf + '\n';
	}
	std::size_t samples = strdata.size() - ctx;

	std::vector<std::vector<float>> inp(samples, std::vector<float>(ctx , 0.f));
	std::vector<std::vector<float>> out(samples, std::vector<float>(255, 0.f));

	for (std::size_t s = 0; s < samples; ++s)
	{
		for (std::size_t c = 0; c < ctx; ++c)
			inp[s][c] = (float)strdata[s + c];
		out[s][(int)strdata[s + ctx + 1]] = 1.f;
	}

	tp[1] = std::chrono::high_resolution_clock::now();
	std::cout << "Parsed the data in " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms.\n";

	// Define the model
	tp[0] = std::chrono::high_resolution_clock::now();

	MLP::MLP vs( { ctx, (ctx + 255) / 2, 255} ); // vs virtual shakespeare
	float lr = 0.05f;
	
	tp[1] = std::chrono::high_resolution_clock::now();
	std::cout << "Defined the model in " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms.\n";

	// Train the model
	tp[0] = std::chrono::high_resolution_clock::now();

	omp_set_num_threads(6); // Speeds training up a little using OMP
	vs.enableTraining(); // Enable training, isn't enabled by default to save some memory
	vs.train(inp, out, sigmoid, sigmoidDV, lr, 2); // Train the model on the data parsed using Sigmoid and two epochs
	vs.disableTraining(); // Disable training after training to save some memory

	tp[1] = std::chrono::high_resolution_clock::now();
	std::cout << "Trained the model in " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms.\n";

	// Save the model
	if (vs.save("VirtualShakespeare.MLP")) std::cout << "Saved Model!\n";
	else std::cout << "Failed to save Model =(\n";

	// Inference it
	std::string input("");
	std::vector<float> inpvec(ctx, 0.f);
	while (input != std::string("---EXIT---"))
	{
		std::cout << "> ";
		std::getline(std::cin, input);

		if (input.size() > ctx) input = input.substr(0, 12);
		for (unsigned short c = 0; c < input.size(); ++c)
			inpvec[c] = (float)input[c];

		auto out = vs.out(inpvec, sigmoid);
		std::cout << "< " << char(std::max_element(out.begin(), out.end()) - out.begin()) << '\n';

		inpvec = std::vector<float>(ctx, 0.f);
	}

	return 0;
}
