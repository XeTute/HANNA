#include <iostream>

#include "HANNA/HANNA.hpp"

// Test 0.00 for HANNA's port to Eigen
// Test-Code licensed under MIT, HANNA Library under Apache2.0, and multi for Eigen
// Port to OpenCL expected in four - x months.

void ReLU(float& x) { x = (x > 0) ? x : 0; }
void ReLUDV(float& x) { x = (x > 0) ? 1.f : x; }

void sigmoid(float& x) { x = 1.f / (1.f + std::exp(-x)); }
void sigmoidDV(float& x) { x *= (1 - x); }

/*
  Current Problem: Output always ~0.49999 for all inputs after training
*/

int main()
{
	// MLP::MLP net({ 2, 3, 1 });
	// omp_set_num_threads(1);
	// 
	// std::vector<std::vector<std::vector<float>>> data =
	// {
	// 	{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
	// 	{ { 0 }, { 1 }, { 1 }, { 0 }}
	// };
	// float lr = 0.1f;
	// PERCEPTRON::n epochs = 50000;
	// 
	// net.enableTraining();
	// net.train(data[0], data[1], sigmoid, sigmoidDV, lr, epochs);
	// net.disableTraining();
	// 
	// for (PERCEPTRON::n s = 0; s < 4; ++s)
	// {
	// 	std::cout << "---\nInput: " << data[0][s][0] << " : " << data[0][s][1] << '\n';
	// 	net.forward(data[0][s], sigmoid);
	// 	std::cout << "Output: " << net.out()[0] << '\n';
	// }
	// 
	// if (!net.save("XORNet.MLP")) std::cout << "Failed";

	MLP::MLP net;
	if (!net.load("XORNet.MLP")) std::cout << "Failed to load. Will props crash.\n";
	omp_set_num_threads(1);

	std::vector<std::vector<std::vector<float>>> data =
	{
		{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
		{ { 0 }, { 1 }, { 1 }, { 0 }}
	};

	for (PERCEPTRON::n s = 0; s < 4; ++s)
	{
		std::cout << "---\nInput: " << data[0][s][0] << " : " << data[0][s][1] << '\n';
		net.forward(data[0][s], sigmoid);
		std::cout << "Output: " << net.out()[0] << '\n';
	}

	return 0;
}