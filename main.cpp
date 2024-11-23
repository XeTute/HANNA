#include <iostream>
#include <chrono>
#include <cmath>

#include "ANNA.hpp"

using namespace std::literals::chrono_literals;
using namespace std::chrono;
using namespace _ANNA;

typedef time_point<high_resolution_clock> Timepoint;
typedef high_resolution_clock hdc; // high res clock

#define PATH "XOR.anna"

int main()
{
	std::vector<counter> scale = { 2, 40, 40, 40, 1 }; // Doesn't have to be that large
	ANNA<float> myNet;
	myNet.setThreads(1);

	myNet.lr = float(0.175);

	std::vector<std::vector<std::vector<float>>> data =
	{
		{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
		{ { 0 }, { 1 }, { 1 }, { 0 } }
	};
	counter md = data[0].size();

	try
	{
		throw std::exception("");
		myNet.load(PATH);
		std::cout << "Loaded model from " << PATH << ".\n";
	}
	catch (...)
	{
		std::cout << "Error occurred while trying to load model from " << PATH << ".\n";
		myNet.init(scale);
	}
	std::cout << "Using " << omp_get_max_threads() << " threads on " << myNet.getNParams() << " Parameters.\n";

	Timepoint timepoint[2] = { hdc::now() };
#pragma omp parallel for collapse(2)
	for (counter e = 0; e < 70000; ++e)
	{
		for (counter s = 0; s < md; ++s)
		{
			myNet.forward(data[0][s]);
			myNet.backward(data[1][s]);
		}
	}
	timepoint[1] = hdc::now();
	std::cout << "Took " << duration_cast<milliseconds>(timepoint[1] - timepoint[0]).count() << "ms to train.\n";

	myNet.save(PATH);
	std::cout << "MSE: " << myNet.getMSE(data);

	return 0;
}
