#include <fstream>
#include <sstream>
#include <vector>
#include <valarray>
#include <string>
#include <cstdint>

namespace DPP
{
	std::vector<std::valarray<float>> readCSV(std::string path)
	{
		std::ifstream r(path, std::ios::in);
		if (!r || !r.is_open() || !r.good()) return std::vector<std::valarray<float>>(0);

		std::vector<std::valarray<float>> data(0);
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
			std::valarray<float> rowvec(elems);

			for (std::size_t i = 0; std::getline(row, elem, ','); ++i)
				rowvec[i] = std::stof(elem);
			data.push_back(rowvec);
		}
		return data;
	}
};