#ifndef MLP_HPP
#define MLP_HPP

#include <cmath>
#include <iostream>
#include "perceptron.hpp"

namespace MLP
{
	class MLP
	{
	private:
		using n = PERCEPTRON::n;

		std::vector<PERCEPTRON::LAYER> perc;
		std::vector<float> last_input;
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

		const void forward(const std::vector<float>& inp, void (*activation) (float&))
		{
			perc[0].forward(inp, activation);
			for (n layer = 1; layer < dl; ++layer)
				perc[layer].forward(perc[layer - 1].getCS(), activation);
		}

		void forwardForGrad(const std::vector<float>& inp, void (*activation) (float&))
		{
			last_input = inp;
			forward(inp, activation);
		}

		const std::vector<float>& out() { return perc[ddl].getCS(); }
		const std::vector<float>& out(const std::vector<float>& inp, void (*activation) (float&))
		{
			forward(inp, activation);
			return out();
		}

		void gradDesc(const std::vector<float>& expected_output, void (*activationDV) (float&), const float& lr)
		{
			perc[ddl].gradDesc(perc[dddl].getCS(), expected_output, activationDV, lr);
			for (n layer = dddl; layer > 0; --layer)
				perc[layer].gradDesc(perc[layer - 1].getCS(), perc[layer + 1].lei(), activationDV, lr);
			perc[0].gradDesc(last_input, perc[1].lei(), activationDV, lr);
		}

		void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& output, void (*activation) (float&), void (*activationDV) (float&), const float& lr, const n& epochs)
		{
			if (input.size() != output.size()) std::cerr << "[HANNA-MLP : MLP::MLP::train([...])]: Input Samples don't match Output Samples. Will train on min samples.\n";
			n samples = std::min(input.size(), output.size());
			for (n e = 0; e < epochs; ++e)
			{
				for (n s = 0; s < samples; ++s)
				{
					forwardForGrad(input[s], activation);
					gradDesc(output[s], activationDV, lr);
				}
			}
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