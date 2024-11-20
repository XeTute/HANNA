#ifndef ANNA_HPP
#define ANNA_HPP
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

// ANNA = Asadullah's Neural Network Architecture

typedef __int64 counter; // Yall can thank OpenMP for not allowing unsigned types

namespace XTTNNC // XeTute Technologies NN Collection
{
	std::random_device rd;
	std::uniform_real_distribution<float> urd(-1.5, 1.5); // urd(rd) will return a "random" number.

	template <typename prec>
	class ANNA
	{
	private:
		typedef std::vector<prec> pa; // pa = prec array
		typedef std::vector<pa> pa2; // pa2 = prec array 2 dim
		typedef std::vector<pa2> pa3; // pa3 = prec array 3 dim

		pa3 weight;
		pa2 neuron_value;
		pa2 neuron_bias;
		pa2 delta;

		std::vector<counter> scale;

		counter layers;
		counter d_layers; // decreased layers, layers - 1
		counter threads;

		prec sigmoid(prec x) { return 1 / (1 + std::exp(-x)); }
		prec sigDeri(prec x)
		{
			prec sig = sigmoid(x);
			return sig * (1 - sig);
		}

	public:
		prec lr = 0.0f;

		ANNA()
		{
			weight = pa3(0);
			neuron_value = pa2(0);
			neuron_bias = pa2(0);
			delta = pa2(0);

			scale = std::vector<counter>(0);

			layers = 0;
			d_layers = 0;
			lr = 0.0f;
			threads = 0;

			std::cout << "ANNA: Prepared for init.\n";
		}

		ANNA(std::vector<counter> _scale) { init(_scale); }
		ANNA(std::vector<counter> _scale, counter _threads) { init(_scale); setThreads(_threads); }

		void setThreads(counter _threads) { omp_set_num_threads((int)_threads); }

		void init(std::vector<counter> _scale)
		{
			if (_scale.size() < 3)
			{
				throw std::runtime_error("ANNA requires at least total of three layers.");
				return;
			}
			scale = _scale;
			layers = scale.size();
			d_layers = layers - 1;

			weight = pa3(d_layers);
			neuron_value = pa2(layers);
			neuron_bias = pa2(d_layers);
			delta = pa2(layers);

			for (counter layer = 0; layer < d_layers; ++layer) // < d_layers for weight
			{
				counter neurons = scale[layer]; // will be used frequently, copy to L cache

				neuron_value[layer] = pa(neurons);
				neuron_bias[layer] = pa(neurons);
				delta[layer] = pa(neurons);

				weight[layer] = pa2(neurons);

				for (counter neuron = 0; neuron < neurons; ++neuron)
				{
					neuron_bias[layer][neuron] = (prec)urd(rd);
					weight[layer][neuron] = pa(scale[layer + 1]);

					for (counter neuron_nl = 0; neuron_nl < scale[layer + 1]; ++neuron_nl)
						weight[layer][neuron][neuron_nl] = (prec)urd(rd);
				}
			}

			neuron_value[d_layers] = pa(scale[d_layers]);
			delta[d_layers] = pa(scale[d_layers]);
			neuron_bias[0] = pa(0);
		}

		counter getNParams()
		{
			counter rv = 0;
			for (counter l = 0; l < d_layers; ++l) rv += scale[l] * scale[l + 1] + scale[l];
			rv -= scale[0];
			rv -= scale[d_layers];
			return rv;
		}

		void save(std::string path)
		{
			std::ofstream w(path, std::ios::binary);
			if (w.is_open())
			{
				for (counter i = 0; i < layers; ++i) w << scale[i] << ';';
				w << '\n';

				size_t sv = sizeof(prec); // sv = size variable
				for (counter neuron = 0; neuron < scale[0]; ++neuron)
					w.write((char*)weight[0][neuron].data(), weight[0][neuron].size() * sv);

				for (counter layer = 1; layer < d_layers; ++layer)
				{
					for (counter neuron = 0; neuron < scale[layer]; ++neuron)
						w.write((char*)weight[layer][neuron].data(), weight[layer][neuron].size() * sv);
					w.write((char*)neuron_bias[layer].data(), neuron_bias[layer].size() * sv);
				}
			}
			else throw std::runtime_error(std::string("Couldn't open " + path + " to save ANNA.").c_str());
		}

		void load(std::string path)
		{
			std::ifstream r(path);
			if (r.is_open())
			{
				{ // Read scale
					std::string buffer[] = { "", "" };
					std::getline(r, buffer[0]);

					counter mb = buffer[0].size();
					scale = std::vector<counter>(0);
					counter ms = scale.size();

					for (counter i = 0; i < mb; ++i)
					{
						if (buffer[0][i] == ';')
						{
							scale.push_back(std::stoull(buffer[1]));
							buffer[1].clear();
						}
						else buffer[1] += buffer[0][i];
					}
					for (counter i = 0; i < ms; ++i) std::cout << scale[i] << '\n';
					this->init(scale);

					r = std::ifstream(path, std::ios::binary);
					std::getline(r, buffer[0]);
				}
				size_t sv = sizeof(prec); // sv = size variable

				try
				{
					for (counter neuron = 0; neuron < scale[0]; ++neuron)
						r.read((char*)weight[0][neuron].data(), weight[0][neuron].size() * sv);

					for (counter layer = 1; layer < d_layers; ++layer)
					{
						for (counter neuron = 0; neuron < scale[layer]; ++neuron)
							r.read((char*)weight[layer][neuron].data(), weight[layer][neuron].size() * sv);
						r.read((char*)neuron_bias[layer].data(), neuron_bias[layer].size() * sv);
					}
				}
				catch (...)
				{
					r.close();
					throw std::runtime_error(std::string("Coudn't open " + path + " to load ANNA.").c_str());
					return;
				}
			}
			else throw std::runtime_error(std::string("Coudn't open " + path + " to load ANNA.").c_str());
		}

		void forward(pa& i) // I assume that input has the same amount of input neurons as the MLP
		{
			neuron_value[0] = i;

			counter ccn; // cached counter

			for (counter l = 1; l < d_layers; ++l)
			{
				counter dl = l - 1;
				ccn = scale[l];
#pragma omp parallel for
				for (counter n = 0; n < ccn; ++n)
				{
					counter mcln = scale[dl];
					neuron_value[l][n] = neuron_bias[l][n];

					for (counter cln = 0; cln < mcln; ++cln) // cln current last neuron
						neuron_value[l][n] += (neuron_value[dl][cln] * weight[dl][cln][n]);
					neuron_value[l][n] = sigmoid(neuron_value[l][n]);
				}
			}

			ccn = scale[d_layers];
			counter ddl = d_layers - 1;
#pragma omp parallel for
			for (counter n = 0; n < ccn; ++n)
			{
				counter mnl = scale[ddl];
				neuron_value[d_layers][n] = prec(0.0f);

				for (counter nl = 0; nl < mnl; ++nl)
					neuron_value[d_layers][n] += (neuron_value[ddl][nl] * weight[ddl][nl][n]);
				neuron_value[d_layers][n] = sigmoid(neuron_value[d_layers][n]);
			}
		}

		pa getOutput() { return neuron_value[d_layers]; }

		void backward(pa& eo) // eo expected output
		{
			counter mn = scale[d_layers]; // mn max neurons
			counter ddl = d_layers - 1;

#pragma omp parallel for
			for (counter n = 0; n < mn; ++n)
			{
				prec ao = neuron_value[d_layers][n]; // caching the current neuron
				delta[d_layers][n] = (eo[n] - ao) * sigDeri(ao);
			}

			for (counter l = ddl; l >= 0; --l)
			{
				counter ml = scale[l];
#pragma omp parallel for
				for (counter n = 0; n < ml; ++n)
				{
					prec e = prec(0.0f); // e error

					counter il = l + 1; // il increased layer
					counter mnn = scale[il];

					for (counter nn = 0; nn < mnn; ++nn) // nn next neuron(in the next layer)
						e += delta[il][nn] * weight[l][n][nn];

					delta[l][n] = e * sigDeri(neuron_value[l][n]);
				}
			}

			// Applying the changes
			mn = scale[d_layers];
#pragma omp parallel for
			for (counter n = 0; n < mn; ++n)
			{
				for (counter nl = 0; nl < ddl; ++nl) // nl neuron last layer
					weight[ddl][nl][n] += neuron_value[ddl][nl] * delta[d_layers][n] * lr;
			}

#pragma omp parallel for
			for (counter l = ddl; l > 0; --l)
			{
				mn = scale[l];
				for (counter n = 0; n < mn; ++n)
				{
					neuron_bias[l][n] += delta[l][n] * lr;

					counter dl = l - 1;
					counter mnl = scale[dl];

					for (counter nl = 0; nl < mnl; ++nl) //nl neuron last layer
						weight[dl][nl][n] += neuron_value[dl][nl] * delta[l][n] * lr;
				}
			}
		}

		prec getMSE(pa3& d) // d[0 == input, 1 == output][sample][...]
		{
			prec mse = 0.0f;
			counter md = d.size();
			pa mo;

			for (counter s = 0; s < md; ++s)
			{
				this->forward(d[0][s]);
				mo = this->getOutput(); // mo model output
				counter mmo = mo.size();
				for (counter n = 0; n < mmo; ++n)
				{
					prec e = d[1][s][n] - mo[n];
					mse += e * e;
				}
			}
			return mse / (md * mo.size());
		}
	};
};
