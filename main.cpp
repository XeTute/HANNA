#include <iostream>

#include "HANNA/HANNA.hpp"

// Test 0.00 for HANNA's port to Eigen
// Test-Code licensed under MIT, HANNA Library under Apache2.0, and multi for Eigen
// Port to OpenCL expected in four - x months.

float activation(float x) { return x; }
int main()
{
	PERCEPTRON::LAYER model;
	if (model.birth(6, 1)) std::cout << "Success\n";
	else std::cout << "Fail\n";
	
	model.suicide();
	return 0;
}