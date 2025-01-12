#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include "Eigen/Dense"

namespace PERCEPTRON
{
	using n = size_t;
	using str = const char*;
	using constVecRef = const Eigen::VectorXf&;

	class LAYER
	{
	private:
		n i, o; // input output, input = neurons last layer, output = neurons this layer

		Eigen::VectorXf val, bias;
		Eigen::VectorXf delta, expected_input;
		Eigen::MatrixXf weight; // Connecting the last and `this` layer.

		void alloc()
		{
			val.resize(o);
			bias.resize(o);
			weight.resize(o, i);
		}

		void randomParams()
		{
			n t = Eigen::nbThreads();
			Eigen::setNbThreads(0);

			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<float> dist(-1.f, 1.f);

			auto randomize = [&]() { return dist(gen); };
			bias = Eigen::VectorXf::NullaryExpr(o, randomize);
			weight = Eigen::MatrixXf::NullaryExpr(o, i, randomize);

			Eigen::setNbThreads(t);
		}
	public:
		LAYER() : i(0), o(0) {}
		LAYER(n input, n output) { birth(input, output); }

		void birth(n input, n output)
		{
			i = input;
			o = output;

			alloc();
			randomParams();
			disableTraining();
		}
		void setThreads(n threads) { Eigen::setNbThreads(threads); }

		void enableTraining()
		{
			delta.resize(o);
			expected_input.resize(i);
		}

		void disableTraining()
		{
			delta.resize(0);
			expected_input.resize(0);
		}

		bool save(str path)
		{
			std::ofstream w(path, std::ios::out | std::ios::binary);
			constexpr n sf = sizeof(float);
			constexpr n sn = sizeof(n);

			if (!w.is_open() || !w) return false;
			if (!w.write((const char*)&i, sn)) return false;
			if (!w.write((const char*)&o, sn)) return false;
			if (!w.write((const char*)weight.data(), sf * o * i)) return false;
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
			if (!r.read((char*)weight.data(), sf * o * i)) return false;
			if (!r.read((char*)bias.data(), sf * o)) return false;

			return true;
		}

		void forward(constVecRef input, float (*activation) (float))
		{
			val = (weight * input) + bias;
			val.unaryExpr(activation);
		}

		void forward(constVecRef input, Eigen::VectorXf& output, float (*activation) (float))
		{
			forward(input, activation);
			output = val;
		}

		constVecRef gradDesc(constVecRef last_input, constVecRef expected_output, float (*activatioDV) (float), const float& lr)
		{
			expected_input.setZero();

			delta = expected_output - val;
			val.unaryExpr(activatioDV);
			delta = delta.array() * val.array();

			bias += delta * lr;
			weight += last_input * delta.transpose() * lr;
			expected_input += delta * weight;

			return expected_input;
		}

		void suicide()
		{
			i = 0, o = 0;

			val.resize(0);
			bias.resize(0);
			weight.resize(0, 0);

			disableTraining();
		}

		~LAYER() { suicide(); }
	};
}

#endif