#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>

#include "HANNA/HANNA.hpp"

void sigmoid(float& x) { x = 1.f / (1.f + std::exp(-x)); }
void sigmoidDV(float& x) { x *= (1.f - x); }

struct normConf { float min, delta; };
void minmaxnorm(std::vector<float>& a, normConf conf)
{
	if (a.empty()) return;
	if (conf.min == *std::max_element(a.begin(), a.end())) std::fill(a.begin(), a.end(), 0.f);

	long long int size = a.size();
#pragma omp parallel for
	for (long long int i = 0; i < size; ++i)
		a[i] = (a[i] - conf.min) / conf.delta;
}

std::vector<std::vector<float>> readCSV(std::string path)
{
	std::ifstream r(path, std::ios::in);
	if (!r || !r.is_open() || !r.good()) return std::vector<std::vector<float>>(0);

	std::vector<std::vector<float>> data(0);
	std::size_t elems = 0;

	std::string buffer("");
	std::string elem("");

	std::getline(r, buffer); // First line is header
	{
		std::stringstream header(buffer);
		while (std::getline(header, elem, ','))
			++elems;
	}

	while (std::getline(r, buffer))
	{
		std::stringstream row(buffer);
		std::vector<float> rowvec(elems);
		
		for (std::size_t i = 0; std::getline(row, elem, ','); ++i)
			rowvec[i] = std::stof(elem);
		data.push_back(rowvec);
	}
	return data;
}

int main()
{
	omp_set_num_threads(6);

	MLP::MLP hdm; // heart disease model
	hdm.birth({ });

	return 0;
}
