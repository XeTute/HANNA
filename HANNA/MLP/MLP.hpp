#ifndef MLP_HPP
#define MLP_HPP

#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <filesystem>
#include "perceptron.hpp"

namespace MLP
{
	class MLP
	{
	private:
		using n = PERCEPTRON::n;

		std::vector<PERCEPTRON::LAYER> perc;
		std::valarray<float> last_input;
		std::vector<n> ls; // layer scale
		n l, dl, ddl, dddl;

		void alloc()
		{
			perc.resize(dl);
			for (n layer = 0; layer < dl; ++layer)
				perc[layer].birth(ls[layer], ls[layer + 1]);
		}

	public:
		MLP() : perc(0), ls(0), l(0), dl(0), ddl(0), dddl(0) {}
		MLP(std::vector<n> _scale) { birth(_scale); }

		void birth(std::vector<n> _scale)
		{
			ls = _scale;

			l = ls.size();
			dl = l - 1;
			ddl = dl - 1;
			dddl = ddl - 1;

			alloc();
		}

		n getNParams()
		{
			n params = 0;
			for (n layer = 1; layer < l; ++layer)
				params += ls[layer] * ls[layer - 1] + ls[layer];
			return params;
		}

		void enableTraining()
		{
			for (n layer = 0; layer < dl; ++layer)
				perc[layer].enableTraining();
			last_input.resize(ls[0]);
		}

		void disableTraining()
		{
			for (n layer = 0; layer < dl; ++layer)
				perc[layer].disableTraining();
			last_input.resize(0);
		}

		const void forward(const std::valarray<float>& inp, void (*activation) (float&))
		{
			perc[0].forward(inp, activation);
			for (n layer = 1; layer < dl; ++layer)
				perc[layer].forward(perc[layer - 1].getCS(), activation);
		}

		void forwardForGrad(const std::valarray<float>& inp, void (*activation) (float&))
		{
			last_input = inp;
			forward(inp, activation);
		}

		const std::valarray<float>& out() { return perc[ddl].getCS(); }
		const std::valarray<float>& out(const std::valarray<float>& inp, void (*activation) (float&))
		{
			forward(inp, activation);
			return out();
		}

		void gradDesc(const std::valarray<float>& expected_output, void (*activationDV) (float&), const float& lr)
		{
			perc[ddl].gradDesc(perc[dddl].getCS(), expected_output, activationDV, lr);
			for (n layer = dddl; layer > 0; --layer)
				perc[layer].gradDesc(perc[layer - 1].getCS(), perc[layer + 1].lei(), activationDV, lr);
			perc[0].gradDesc(last_input, perc[1].lei(), activationDV, lr);
		}

		void train
		(
			const std::vector<std::valarray<float>>& input,
			const std::vector<std::valarray<float>>& output,
			void (*activation) (float&),
			void (*activationDV) (float&),
			float lr,
			float multby,
			const n& epochs
		)
		{
			if (input.size() != output.size()) std::cerr << "[HANNA-MLP : MLP::MLP::train([...])]: Input Samples don't match Output Samples. Will train on min samples.\n";
			n samples = std::min(input.size(), output.size());
			enableTraining();
			for (n e = 0; e < epochs; ++e)
			{
				for (n s = 0; s < samples; ++s)
				{
					forwardForGrad(input[s], activation);
					gradDesc(output[s], activationDV, lr);
				}
				lr *= multby;
			}
			disableTraining();
		}

		bool save(PERCEPTRON::str path)
		{
			std::filesystem::create_directory(path);
			std::ofstream w(std::string(path) + std::string("/DNA.MLP"), std::ios::out | std::ios::binary);
			if (!w || !w.is_open()) return false;

			if ( !w.write((const char*)&l, sizeof(l)) ) return false;
			if ( !w.write((const char*)ls.data(), sizeof(ls[0]) * ls.size()) ) return false;
			w.close();

			std::string tmppath = std::string(path) + std::string("/LAYER");
			for (n layer = 0; layer < dl; ++layer)
				if ( !perc[layer].save((tmppath + std::to_string(layer) + std::string(".MLP")).c_str()) ) return false;
			return true;
		}

		bool load(PERCEPTRON::str path)
		{
			std::ifstream r(std::string(path) + std::string("/DNA.MLP"), std::ios::in | std::ios::binary);
			if (!r || !r.is_open()) return false;

			if ( !r.read((char*)&l, sizeof(l)) ) return false;
			ls.resize(l);
			dl = l - 1;
			ddl = dl - 1;
			dddl = ddl - 1;
			if ( !r.read((char*)ls.data(), sizeof(ls[0]) * ls.size()) ) return false;
			alloc();

			std::string tmppath = std::string(path) + std::string("/LAYER");
			for (n layer = 0; layer < dl; ++layer)
				if ( !perc[layer].load((tmppath + std::to_string(layer) + std::string(".MLP")).c_str()) ) return false;
			return true;
		}

		~MLP()
		{
			perc.resize(0);
			ls.resize(0);
			l = 0, dl = 0, ddl = 0, dddl = 0;
		}
	};
}
#endif
