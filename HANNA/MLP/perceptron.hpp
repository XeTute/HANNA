#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <fstream>
#include <omp.h>
#include <random>
#include <vector>

namespace PERCEPTRON
{
	using n = long long int; // Thank omp for not allowing signed
	using str = const char*;

	class LAYER
	{
	private:
		n i, o; // input output, input = neurons last layer, output = neurons this layer

		std::vector<float> val, bias;
		std::vector<float> expected_input;
		std::vector<std::vector<float>> weight; // Connecting the last and `this` layer.

		bool alloc()
		{
			val.resize(o);
			bias.resize(o);
			weight.resize(o);
			for (std::vector<float>& vec : weight)
				vec.resize(i);
			return true;
		}

		void random()
		{
			std::mt19937 gen(std::random_device{}());
			std::uniform_real_distribution<float> dist(-1.f, 1.f);

			for (float& x : bias) x = dist(gen);
			for (std::vector<float>& vec : weight)
				for (float& x : vec) x = dist(gen);
		}

	public:
		LAYER() : i(0), o(0), val(0), bias(0), expected_input(0), weight(0) {}
		LAYER(n input, n output) { birth(input, output); }

		void enableTraining()
		{
			expected_input.resize(i);
		}

		void disableTraining()
		{
			expected_input.resize(0);
		}

		bool birth(n input, n output)
		{
			i = input, o = output;
			if (!alloc()) return false;
			random();
			return true;
		}

		void suicide()
		{
			i = 0, o = 0;
			val.resize(0);
			bias.resize(0);
			for (n neuron = 0; neuron < o; ++neuron)
				weight[o].resize(0);
		}

		bool save(str path)
		{
			std::ofstream w(path, std::ios::out | std::ios::binary);
			constexpr n sf = sizeof(float);
			constexpr n sn = sizeof(n);

			if (!w.is_open() || !w) return false;
			if (!w.write((const char*)&i, sn)) return false;
			if (!w.write((const char*)&o, sn)) return false;
			for (std::vector<float>& vec : weight)
				if (!w.write((const char*)vec.data(), sf * i)) return false;
			if (!w.write((const char*)bias.data(), sf * o)) return false;

			return true;
		}

		bool load(str path)
		{
			std::ifstream r(path, std::ios::in | std::ios::binary);
			constexpr n sf = sizeof(float);
			constexpr n sn = sizeof(n);

			if (!r.is_open() || !r) return false;
			if (!r.read((char*)&i, sn)) return false;
			if (!r.read((char*)&o, sn)) return false;

			alloc();
			for (std::vector<float>& vec : weight)
				if (!r.read((char*)vec.data(), sf * i)) return false;
			if (!r.read((char*)bias.data(), sf * o)) return false;

			return true;
		}

		void forward(const std::vector<float>& input, void (*activation) (float&))
		{
#pragma omp parallel for
			for (n neuron = 0; neuron < o; ++neuron)
			{
				float tmpval = bias[neuron];
				for (n nll = 0; nll < i; ++nll)
					tmpval += weight[neuron][nll] * input[nll];
				activation(tmpval);
				val[neuron] = tmpval;
			}
		}
		const std::vector<float>& getCS() { return val; } // CS current state

		const std::vector<float>& gradDesc(const std::vector<float>& last_input, const std::vector<float>& expected_output, void (*activationDV) (float&), const float& lr)
		{
#pragma omp parallel for
			for (n neuron = 0; neuron < o; ++neuron)
			{
				float tmpval = val[neuron];
				float delta = expected_output[neuron] - tmpval;
				activationDV(tmpval);
				delta *= tmpval;

				float deltalr = delta * lr;
				bias[neuron] += deltalr;

				for (n nll = 0; nll < i; ++nll)
				{
					weight[neuron][nll] += deltalr * last_input[nll];
					expected_input[nll] += delta * weight[neuron][nll];
				}
			}

			return expected_input;
		}
		const std::vector<float>& lei() { return expected_input; } // last expected input

		~LAYER() { suicide(); }
	};
}

#endif