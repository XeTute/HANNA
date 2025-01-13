#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <fstream>

#include "whitebox.hpp"

namespace PERCEPTRON
{
	using wb::n;
	using str = const char*;
	using constVecRef = const wb::effarr<float>&;

	class LAYER
	{
	private:
		n i, o; // input output, input = neurons last layer, output = neurons this layer

		wb::effarr<float> val, bias;
		wb::effarr<float> delta, expected_input;
		wb::effarr<wb::effarr<float>> weight; // Connecting the last and `this` layer.

		bool alloc()
		{
			if (!val.resize(o)) return false;
			if (!bias.resize(o)) return false;
			if (!weight.resize(o)) return false;
			for (n neuron = 0; neuron < o; ++neuron)
			{
				weight[neuron].forceResetWithoutFree();
				if (!weight[neuron].resize(i)) return false;
			}
			return true;
		}

		void random()
		{
			val.random();
			bias.random();
			for (n neuron = 0; neuron < o; ++neuron)
				weight[neuron].random();
		}

	public:
		LAYER() : i(0), o(0), val(0), bias(0), delta(0), expected_input(0), weight(0) {}
		LAYER(n input, n output) { birth(input, output); }

		bool enableTraining()
		{
			if (!delta.resize(o)) return false;
			if (!expected_input.resize(i)) return false;
			return true;
		}

		void disableTraining()
		{
			delta.resize(0);
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
			for (n neuron = 0; neuron < o; ++neuron)
				if (!w.write((const char*)weight[neuron].data(), sf * i)) return false;
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
			for (n neuron = 0; neuron < o; ++neuron)
				if (!r.read((char*)weight[neuron].data(), sf * i)) return false;
			if (!r.read((char*)bias.data(), sf * o)) return false;

			return true;
		}

		~LAYER() { suicide(); }
	};
}

#endif