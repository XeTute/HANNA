#include <iostream>
#include <chrono>
#include <cmath>
#include <string>

#include "ANNA.hpp"

using namespace std::literals::chrono_literals;
using namespace std::chrono;
using namespace _ANNA;

typedef time_point<high_resolution_clock> Timepoint;
typedef high_resolution_clock hdc; // high res clock

constexpr auto PATH = "HD.anna"; // HDD heart disease

int main()
{
	ANNA<float> HDN; // HDN heart disease network
	HDN.setThreads(6);
	HDN.lr = float(0.01); // 0.001 ~ (1 / 1025) (1025 is the amount of samples in the training data)
	
	try
	{
		HDN.load(PATH);
		std::cout
			<< "Loaded Heart Disease Network from file " << PATH
			<< ".\nIt contains " << HDN.getNParams() << " parameters.\n";
	}
	catch (...)
	{
		HDN.init({13, 21, 17, 17, 9, 3, 1});
		std::cout
			<< "Failed to load Heart Disease Network from file " << PATH
			<< ".\nTraining from scratch, the new network contains " << HDN.getNParams() << " parameters.\n";
	}

	std::vector<std::vector<std::vector<float>>> d(2, std::vector<std::vector<float>>()); // d dataset, d[0], input d[1] output

	{ // Format dataset
		std::vector<std::vector<float>> csv = loadCSVf("hdd.csv"); // loadCSVf ( f = float ) in _ANNA::

		std::size_t ms = csv.size(); // ms max samples
		std::size_t me = csv[0].size(); // me max elems per sample
		std::size_t dme = csv[0].size() - 1;

		d[0] = std::vector<std::vector<float>>(ms, std::vector<float>(me, 0.0f));
		d[1] = std::vector<std::vector<float>>(ms, std::vector<float>(1, 0.0f));

		for (std::size_t s = 0; s < ms; ++s)
		{
			for (std::size_t e = 0; e < me; ++e) d[0][s][e] = csv[s][e];
		}

		csv.clear();
	}

	std::size_t ms = d[0].size();

	Timepoint tp[2] = { hdc::now() };
	std::cout << "Training...\n";

	float mse = 0.0f;
	for (std::size_t e = 0; (e < 103 || mse > 0.15); ++e) // 103 ~ (1025 / 10)
	{
		for (std::size_t s = 0; s < ms; ++s)
		{
			HDN.forward(d[0][s]);
			HDN.backward(d[1][s]);
		}
	}
	tp[1] = hdc::now();

	std::cout
		<< "\nTraining took " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms.\n"
		<< "\nTraining took " << std::chrono::duration_cast<std::chrono::seconds>(tp[1] - tp[0]).count() << "s.\n"
		<< "Training stopped at a MSE(Mean Squared Error) of " << HDN.getMSE(d) << ".\n"
		<< "Will attempt to save the model now...\n";

	try
	{
		HDN.save(PATH);
		std::cout << "Successfully saved the trained model under " << PATH << ".\n";
	}
	catch (...) { std::cout << "Unable to save the model under " << PATH << " =(.\n"; }

	HDN.forward(d[0][0]);
	std::cout
		<< "--- TEST INFERENCE ---\n"
		<< "Given the input of: \n"
		<< "- Age: " << d[0][0][0]
		<< "\n- Sex: " << (d[0][0][1] ? "Male" : "Female")
		<< "\n- Chest Pain Type: " << d[0][0][2]
		<< "\n- Resting Blood Pressure (in mm Hg on admission to the hospital): " << d[0][0][3]
		<< "\n- Serum Cholestoral in mg/dl: " << d[0][0][4]
		<< "\n- Fasting Blood Sugar?: " << (d[0][0][5] ? "Yes" : "No")
		<< "\n- Resting Electrocardiographic Results: " << d[0][0][6]
		<< "\n- Maximum Heart Rate Achieved: " << d[0][0][7]
		<< "\n- Exercise Induced Angina?: " << (d[0][0][8] ? "Yes" : "No")
		<< "\n- ST Depression induced by Exercise relative to Rest: " << d[0][0][9]
		<< "\n- Slope of the Peak Exercise ST Segment: " << d[0][0][10]
		<< "\n- Number of Major Vessels(0-3) colored by Flourosopy: " << d[0][0][11]
		<< "\n- Thal(1: Normal, 2: Fixed Defect, 3: Reversable Defect): " << d[0][0][12]
		<< "\nThe models output was: " << HDN.getOutput()[0]
		<< "\nThe correct output is: " << d[1][0][0] << '\n';

	return 0;
}
