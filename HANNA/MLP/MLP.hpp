#include <cmath>
#include <cstdint>
#include <Eigen/Eigen>
#include <exception>
#include <fstream>
#include <random>
#include <vector>

using un = std::size_t;   // un unsigned number
using sn = std::intmax_t; // sn signed   number

namespace MLP
{
	class LAYER
	{
		Eigen::VectorXf neuron;

		un nrns, nrnsll; // nrnrs neurons, nrnsll neurons last layer

	public:

		Eigen::VectorXf bias;
		Eigen::MatrixXf weight;

		LAYER() : neuron(), bias(), weight(), nrns(0), nrnsll(0) {};
		LAYER(un neurons, un neuronslastlayer) { create(neurons, neuronslastlayer); }

		void create(un neurons, un neuronslastlayer)
		{
			nrns = neurons;
			nrnsll = neuronslastlayer;

			weight = Eigen::MatrixXf(nrns, nrnsll);
			neuron = Eigen::VectorXf(nrns);
			bias = Eigen::VectorXf(nrns);
		}

		void random() // Xavier
		{
			float range = std::sqrtf(6.f / nrns + nrnsll);
			std::mt19937 gen(std::random_device{}());
			std::uniform_real_distribution<float> dist(-range, range);

			for (un nrn = 0; nrn < nrns; ++nrn)
			{
				bias(nrn) = dist(gen);
				for (un nrnll = 0; nrnll < nrnsll; ++nrnll)
					weight.row(nrn)(nrnll) = dist(gen);
			}
		}

		void forward(const Eigen::VectorXf& input, float (*activation)(const float&))
		{
			/*
			neuron & bias: one-dim vector of nrns size
			input: one-dim vector of nrns last layer size
			weight: two-dim matrix of nrns x nrns last layer size

            the line below verbose would be:
			
			for (un nrn = 0; nrn < nrns; ++nrn)
				neuron[nrn] = bias[nrn];
				for (un nrnll = 0; nrnll < nrnsll; ++nrnll)
					neuron[nrn] += weight[nrn][nrnll] * input[nrnll];

			Kinda does a nested loop-ish thing (mentally, I know about allat SIMD & omp), and that's mathematically semicorrect (as far as I know).
			*/

			neuron = (weight * input + bias).unaryExpr(activation);
		}

		Eigen::VectorXf& report() { return neuron; }
		const Eigen::VectorXf report(const Eigen::VectorXf& input, float (*activation) (const float&))
		{
			forward(input, activation);
			return report();
		}

		void graddesc(const Eigen::VectorXf& nexterror, const Eigen::VectorXf& lastinput, float (*activationdr) (const float&), float lr)
		{
			// First Backprop \\ Calculate Errors
			neuron = neuron.unaryExpr(activationdr);
			neuron = nexterror.cwiseProduct(neuron);
			neuron *= lr;

			// Then Gradient Descent \\ Apply Calculated Errors
			bias -= neuron;
			weight -= neuron * lastinput.transpose();
		}

		Eigen::VectorXf getdelta_fll() const // fll for last layer
		{ return weight.transpose() * neuron; }

		void destruct() { create(0, 0); }
		~LAYER() { destruct(); }
	};

	class MLP
	{
		std::vector<LAYER> lyr; // Note: lyr[0] is inactive; just a placeholder to account for the input "layer"
		std::vector<un> nrns;
		un lyrs;

	public:

		std::exception lastexception;

		MLP() : lyr(), nrns(0), lyrs(0) {}
		MLP(std::vector<un> neurons) { birth(neurons); }

		void birth(std::vector<un> neurons)
		{
			lyrs = neurons.size();
			lyr.resize(lyrs);
			nrns = neurons;

			lyr[0].destruct();
			for (un l = 1; l < lyrs; ++l)
				lyr[l].create(nrns[l], nrns[l - 1]);
		}

		bool save(std::string filename)
		{
			try
			{
				std::ofstream w(filename, std::ios::binary);
				if (!w.is_open()) throw std::runtime_error("Failed to open file \"" + filename + "\".");
				un datasize = sizeof(lyr[1].report()[0]);

				w.write((char*)&lyrs, sizeof(un));
				w.write((char*)&nrns[0], lyrs * sizeof(un));

				for (un l = 1; l < lyrs; ++l)
				{
					w.write((char*)&lyr[l].bias[0], nrns[l] * datasize);
					w.write((char*)&lyr[l].weight(0, 0), nrns[l] * nrns[l - 1] * datasize);
				}
				return true;
			}
			catch (const std::exception& e) { lastexception = e; return false; }
		}

		bool load(std::string filename)
		{
			try
			{
				std::ifstream r(filename, std::ios::binary);
				if (!r.is_open()) throw std::runtime_error("Failed to open file \"" + filename + "\".");
				un datasize = sizeof(lyr[1].report()[0]);

				r.read((char*)&lyrs, sizeof(un));
				nrns.resize(lyrs); r.read((char*)&nrns[0], lyrs * sizeof(un));
				birth(nrns);

				for (un l = 1; l < lyrs; ++l)
				{
					r.read((char*)&lyr[l].bias[0], nrns[l] * datasize);
					r.read((char*)&lyr[l].weight(0, 0), nrns[l] * nrns[l - 1] * datasize);
				}
				return true;
			}
			catch (const std::exception& e) { lastexception = e; return false; }
		}

		un get_param_count()
		{
			un params = 0;
			for (un l = 1; l < lyrs; ++l)
				params += nrns[l] + nrns[l] * nrns[l - 1];
			return params;
		}

		void random()
		{
			for (un l = 1; l < lyrs; ++l)
				lyr[l].random();
		}

		void forward(const Eigen::VectorXf& input, float (*activation) (const float&))
		{
			lyr[1].forward(input, activation);
			for (un l = 2; l < lyrs; ++l)
				lyr[l].forward(lyr[l - 1].report(), activation);
		}

		const Eigen::VectorXf& report() { return lyr[lyrs - 1].report(); }
		const Eigen::VectorXf& report(const Eigen::VectorXf& input, float (*activation) (const float&))
		{
			forward(input, activation);
			return lyr[lyrs - 1].report();
		}

		void graddesc(const Eigen::VectorXf& lastinput, const Eigen::VectorXf& expectedoutput, float (*activationdr) (const float&), float lr)
		{
			un mintwo = lyrs - 2;
			lyr[lyrs - 1].graddesc(lyr[lyrs - 1].report() - expectedoutput, lyr[mintwo].report(), activationdr, lr);
			for (un l = mintwo; l > 1; --l)
				lyr[l].graddesc(lyr[l + 1].getdelta_fll(), lyr[l - 1].report(), activationdr, lr);
			lyr[1].graddesc(lyr[2].getdelta_fll(), lastinput, activationdr, lr);
		}

		void kill()
		{
			for (un l = 1; l < lyrs; ++l)
				lyr[l].destruct();
			nrns = std::vector<un>{ 0 };
			lyrs = 0;
		}
		
		~MLP() { kill(); }
	};
}
