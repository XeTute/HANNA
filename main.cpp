#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>

#include "HANNA/HANNA.hpp"

void sigmoid(float& x) { x = 1.f / (1.f + std::exp(-x)); }
void sigmoidDV(float& x) { x *= (1.f - x); }

struct normConf { float min, delta; }; // delta = max - min
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
	hdm.birth({ 12, 7, 7, 3, 1 });

	auto data = readCSV("cardio-hdd.csv");

	std::size_t rows = data.size();
	std::size_t cols = data[0].size();
	std::size_t inputs = cols - 1;

	std::vector<std::vector<float>> input(rows, std::vector<float>(inputs));
	std::vector<std::vector<float>> output(rows, std::vector<float>(1));

	std::vector<normConf> conf(inputs, { 0.f, 0.f });

	for (std::size_t col = 0; col < inputs; ++col)
	{
		std::vector<float> coldata(rows);
#pragma omp parallel for
		for (long long int row = 0; row < rows; ++row)
			coldata[row] = data[row][col];

		float min = *std::min_element(coldata.begin(), coldata.end());
		conf[col] = { min, *std::max_element(coldata.begin(), coldata.end()) - min };
		minmaxnorm(coldata, conf[col]);

#pragma omp parallel for
		for (long long int row = 0; row < rows; ++row)
			data[row][col] = coldata[row];
	}
	std::cout << "Normalized data.\n";

	for (std::size_t row = 0; row < rows; ++row)
	{
		for (std::size_t col = 0; col < inputs; ++col)
			input[row][col] = data[row][col];
		output[row][0] = data[row][inputs];
	}

	data.clear();
	std::cout << "Formatted data.\n";

	float lr = 0.05f;
	std::size_t epochs = 100;
	std::chrono::high_resolution_clock::time_point tp[2];

	hdm.enableTraining();
	tp[0] = std::chrono::high_resolution_clock::now();
	hdm.train(input, output, sigmoid, sigmoidDV, lr, epochs);
	tp[1] = std::chrono::high_resolution_clock::now();
	hdm.disableTraining();

	std::cout << "Took " << std::chrono::duration_cast<std::chrono::seconds>(tp[1] - tp[0]).count() << "s.\n";

	hdm.save("HDM.MLP");
	hdm.forward(input[0], sigmoid);
	std::cout << "Output: " << hdm.out()[0] << '.' << std::endl;

	return 0;
}
