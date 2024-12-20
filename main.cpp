#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

#include <chrono>

#include "ANNA-CPU.hpp"

#ifdef ANNA_OpenCL_HPP
using namespace _ANNA_OCL;
#endif

#ifdef ANNA_CPU_HPP
using namespace _ANNA;
#endif

using namespace std::chrono_literals;

constexpr auto PATH = "cardio-hdd.ANNA";

int main()
{
	ANNA<float> model({ 11, 64, 32, 32, 8, 1});
	model.lr = float(0.1);

	std::vector<std::vector<float>> CSV = loadCSVf("cardio-hdd.csv");
	std::vector<std::vector<std::vector<float>>> train_data(2);

	counter dataset_size = CSV.size();
	counter sample_size = CSV[0].size();
	counter dss = sample_size - 1; // dss is short for decreased sample size

	train_data[0] = std::vector<std::vector<float>>(dataset_size, std::vector<float>(sample_size - 2, 0.f)); // train_data[0] is input
	train_data[1] = std::vector<std::vector<float>>(dataset_size, std::vector<float>(1, 0.f)); // train_data[1] is output

	// Format dataset
	for (counter i = 0; i < dataset_size; ++i)
	{
		for (counter j = 1; j < dss; ++j) train_data[0][i][j - 1] = CSV[i][j];
		train_data[1][i][0] = CSV[i][dss];
	}

	try { model.load(PATH); }
	catch (...) { std::cout << "No Model found saved on " << PATH << ".\n"; }
	std::cout << "Model has a total of " << model.getNParams() << " parameters.\n";

	model.setThreads(6);

	float mse = model.getMSE(train_data); // This will work because train_data[0 = input, 1 = expected output][sample]
	
	std::chrono::time_point<std::chrono::high_resolution_clock> time[2] = { std::chrono::high_resolution_clock::now() };
	
	if (mse >= 0.2) model.train(train_data, 50); // Threaded, took 35s with .setThreads(6) on AMD Ryzen 5 5600G on Windows 10

	// for (counter e = 0; e < 250; ++e) // Single Threaded, took 155s with CPU above with OS above
	// {
	// 	  for (counter s = 0; s < train_data[0].size(); ++s)
	// 	  {
	// 	     model.forward(train_data[0][s]);
	// 	     model.backward(train_data[1][s]);
	// 	  }
	// }

	time[1] = std::chrono::high_resolution_clock::now();

	std::cout << "Took " << std::chrono::duration_cast<std::chrono::seconds>(time[1] - time[0]).count() << "s.\n";

	mse = model.getMSE(train_data);
	std::cout << "MSE after training: " << mse << '\n';

	model.save(PATH);

	return 0;
}
