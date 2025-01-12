#include <iostream>

#include "HANNA/HANNA.hpp"

// Test 0.00 for HANNA's port to Eigen
// Test-Code licensed under MIT, HANNA Library under Apache2.0, and multi for Eigen
// Port to OpenCL expected in four - x months.

float activation(float x) { return x; }
int main()
{
	PERCEPTRON::LAYER model;
	model.birth(6, 1);
	model.setThreads(6);

	Eigen::Vector<float, 6> input = { 10, 20, 30, 40, 50, 60 };
	Eigen::VectorXf output;

	model.forward(input, output, activation);
	std::cout << output << '\n';

	model.suicide();
	return 0;
}