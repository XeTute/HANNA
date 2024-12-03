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

constexpr auto PATH = "HDN.ANNA";

int main()
{
	ANNA<float> HDN; // HDN heart disease network
	HDN.lr = float(0.001);

	try
	{
		HDN.load(PATH);
		std::cout
			<< "Loaded Heart Disease Network from file " << PATH
			<< ".\nIt contains " << HDN.getNParams() << " parameters.\n";
	}
	catch (...)
	{
		HDN.init({ 11, 64, 32, 16, 8, 8, 1 });
		std::cout
			<< "Failed to load Heart Disease Network from file " << PATH
			<< ".\nTraining from scratch, the new network contains " << HDN.getNParams() << " parameters.\n";
	}
	HDN.setThreads(6);

	std::vector<std::vector<std::vector<float>>> d(2, std::vector<std::vector<float>>()); // d dataset, d[0], input d[1] output

	{ // Format dataset
		std::vector<std::vector<float>> csv = loadCSVf("cardio-hdd.csv"); // loadCSVf ( f = float ) in _ANNA::

		std::size_t ms = (16 * 1024); // ms max samples, usually csv.size()
		std::size_t me = csv[0].size(); // me max elems per sample
		std::size_t dme = me - 1;

		d[0] = std::vector<std::vector<float>>(ms, std::vector<float>(dme, 0.0f));
		d[1] = std::vector<std::vector<float>>(ms, std::vector<float>(1, 0.0f));

		for (std::size_t s = 0; s < (16 * 1024); ++s)
		{
			for (std::size_t e = 1; e < dme; ++e) d[0][s][e] = csv[s][e]; // d[0][s][0] is id, not relevant
			d[1][s][0] = csv[s][dme];
		}

		csv.clear();
	}

	std::size_t ms = d[0].size();

	float mse = HDN.getMSE(d);
	std::cout << "Current MSE: " << mse << '\n';
	while (mse > 0.2)
	{
		Timepoint tp[2] = { hdc::now() };
	
		float mse = 1.0f;
		HDN.train(d, 150);
		mse = HDN.getMSE(d);
	
		if (mse > 0.23) HDN.lr = float(0.01);
		else HDN.lr = float(0.001);
	
		tp[1] = hdc::now();
		std::cout
			<< "150 Epochs took " << std::chrono::duration_cast<std::chrono::seconds>(tp[1] - tp[0]).count() << "s.\n"
			<< "MSE: " << mse << '\n';
	
		HDN.save(PATH);
	}

	counter error = 0;
	for (std::size_t s = 0; s < d[0].size(); ++s)
	{
		HDN.forward(d[0][s]);

		// std::cout
		// 	<< "\nGiven the input of: \n"
		// 	<< "- Age(Days): " << d[0][s][0]
		// 	<< "\n- Gender: " << ((d[0][s][1] - 1) ? "Male" : "Female")
		// 	<< "\n- Height(cm): " << d[0][s][2]
		// 	<< "\n- Weight(kg): " << d[0][s][3]
		// 	<< "\n- Systolic blood pressure(mmHg): " << d[0][s][4]
		// 	<< "\n- Diastolic blood pressure(mmHg): " << d[0][s][5]
		// 	<< "\n- Cholesterol: " << ((d[0][s][6] == 1) ? "Normal" : ((d[0][s][6] == 2) ? "Above Normal" : "Well Above Normal"))
		// 	<< "\n- Glucose: " << ((d[0][s][7] == 1) ? "Normal" : ((d[0][s][7] == 2) ? "Above Normal" : "Well Above Normal"))
		// 	<< "\n- Smokes?: " << (d[0][s][8] ? "Yes" : "No")
		// 	<< "\n- Drinks Alcohol?: " << (d[0][s][9] ? "Yes" : "No")
		// 	<< "\n- Physical Activity?: " << (d[0][s][10] ? "Yes" : "No")
		// 	<< "\nThe model predicated: " << ((HDN.getOutput()[0] >= 0.44) ? "Heart Disease" : "No Heart Disease") << " : " << (((HDN.getOutput()[0] >= 0.44) == d[1][s][0]) ? "Correct\n" : "Incorrect\n");

		if (static_cast<int>(HDN.getOutput()[0] >= 0.5) != d[1][s][0]) ++error;
	}
	std::cout << "\nWrong Predictions: " << error << "\nCorrect Predictions: " << d[0].size() - error << '\n';

	return 0;
}
