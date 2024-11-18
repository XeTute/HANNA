#include <iostream>
#include <chrono>
#include <cmath>

#include "ANNA.hpp"

using namespace std::literals::chrono_literals;
using namespace std::chrono;
using namespace XTTNNC;

typedef time_point<high_resolution_clock> Timepoint;
typedef high_resolution_clock hdc; // high res clock

#define PATH "XOR.anna"

int main()
{
	std::vector<counter> scale = { 2, 20, 10, 5, 1 }; // Doesn't have to be that large, but will 100% learn correctly.
	ANNA<float> myNet;
	myNet.setThreads(6);

	std::cout << "Using " << omp_get_max_threads() << " threads.\n";
	
	myNet.lr = float(0.175);

	std::vector<std::vector<std::vector<float>>> data =
	{
		{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
		{ { 0 }, { 1 }, { 1 }, { 0 } }
	};
	counter md = data[0].size();

	try
	{
		myNet.load(PATH);
		std::cout << "Loaded model from " << PATH << ".\n";
	}
	catch (const std::exception& e)
	{
		std::cout << "Error occurred while trying to load model from " << PATH << ": " << e.what() << ".\n";
		myNet.init(scale);
	}
	

	Timepoint timepoint[2] = { hdc::now() };
	for (counter e = 0; e < 7000; ++e)
	{
		for (counter s = 0; s < md; ++s)
		{
			myNet.forward(data[0][s]);
			myNet.backward(data[1][s]);
		}
	}
	timepoint[1] = hdc::now();
	std::cout << "Took " << duration_cast<milliseconds>(timepoint[1] - timepoint[0]).count() << "ms to train.\n";

	float mse = 0.0f;
	for (counter s = 0; s < md; ++s)
	{
		myNet.forward(data[0][s]);
		float o = myNet.getOutput()[0];
		mse = std::pow((data[1][s][0] - o), 2);
		std::cout << std::string(std::to_string(data[0][s][0]) + " | " + std::to_string(data[0][s][1]) + " || " + std::to_string(o)) << '\n';
	}
	std::cout << "MSE: " << mse << '\n';

	myNet.save(PATH);

	return 0;
}
