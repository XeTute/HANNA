#include <iostream>
#include <chrono>

#include "ANNA.hpp"

using namespace std::literals::chrono_literals;
using namespace std::chrono;
using namespace XTTNNC;

typedef time_point<high_resolution_clock> Timepoint;
typedef high_resolution_clock hdc; // high res clock

int main()
{
	std::vector<counter> scale = { 2, 20, 10, 5, 1 }; // Doesn't have to be that large, but will 100% learn correctly.
	ANNA<float> myNet(scale, 6);

	Timepoint timepoint[2] = { hdc::now() };

	std::cout << "Using " << omp_get_max_threads() << " threads.\n";

	try
	{
		myNet.load("myNet.anna");
		std::cout << "Loaded prev. ANNA model.\n";
	}
	catch (const std::exception& e) { std::cout << "Error: " << e.what() << "\nDidn't load model - Will train one from scratch.\n"; }
	
	myNet.lr = float(0.175);

	std::vector<std::vector<std::vector<float>>> data =
	{
		{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
		{ { 0 }, { 1 }, { 1 }, { 0 } }
	};
	counter md = data[0].size();

	for (counter e = 0; e < 10000; ++e)
	{
		for (counter s = 0; s < md; ++s)
		{
			myNet.forward(data[0][s]);
			myNet.backward(data[1][s]);
		}

		if (((e + 1) % 1000) == 0)
		{
			std::cout << "\n--- EPOCH " + std::to_string(e + 1) + " ---\n";
			for (counter s = 0; s < md; ++s)
			{
				myNet.forward(data[0][s]);
				std::cout << std::string(std::to_string(data[0][s][0]) + " | " + std::to_string(data[0][s][1]) + " || " + std::to_string(myNet.getOutput()[0])) << '\n';
			}
		}
	}

	myNet.save("myNet.anna");
	std::cout << "Saved ANNA model.\n";

	timepoint[1] = hdc::now();
	std::cout << "Took " << duration_cast<milliseconds>(timepoint[1] - timepoint[0]).count() << "ms.\n";
	return 0;
}
