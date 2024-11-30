#ifndef ANNA_HPP
#define ANNA_HPP
#endif

#include <cmath> // sqrt
#include <cstdint> // int64_t, uin64_t
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ANNA = Asadullah's Neural Network Architecture

namespace _ANNA
{
	typedef uint64_t counter;
	std::random_device rd;
	std::mt19937 gen(rd()); // gen(rd) will return a "random" number.

	std::vector<std::vector<float>> loadCSVf(std::string path)
	{
		std::ifstream r(path);
		if (!r.is_open())
		{
			throw std::runtime_error(std::string("Failed to load CSV from " + path + '.').c_str());
			return std::vector<std::vector<float>>();
		}

		{
			std::string tmp("");
			std::getline(r, tmp); // first line is definition
		}
		std::vector<std::vector<float>> converted;

		while (!r.eof())
		{
			std::vector<float> push_value(0, 0);
			std::string b[2];
			std::getline(r, b[0]);

			std::size_t mb0 = b[0].size();
			for (std::size_t i = 0; i < mb0; ++i)
			{
				if (b[0][i] == ',')
				{
					push_value.push_back(std::stof(b[1]));
					b[1].clear();
				}
				else b[1] += b[0][i];
			}
			push_value.push_back(std::stof(b[1]));

			converted.push_back(push_value);
		}

		return converted;
	}

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
		counter ddl;

		counter threads;

		counter inp_size;
		counter out_size;

		prec sigmoid(prec x)
		{
			x = std::min(std::max(x, -20.0f), 20.0f); // prevent NaN, inf or other weird numbers
			return 1 / (1 + std::exp(-x));
		}
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
			ddl = 0;
			lr = 0.0f;
			threads = 1;

			std::cout << "ANNA: Prepared for init.\n";
		}

		ANNA(std::vector<counter> _scale) { init(_scale); }
		ANNA(std::vector<counter> _scale, counter _threads) { init(_scale); setThreads(_threads); }

		void setThreads(counter _threads) { threads = _threads; }

		void init(std::vector<counter> _scale)
		{
			if (_scale.size() < 3)
			{
				throw std::runtime_error("ANNA requires at least a total of three layers.");
				return;
			}
			scale = _scale;
			setThreads(1);

			layers = scale.size();
			d_layers = layers - 1;
			ddl = d_layers - 1;

			inp_size = _scale[0];
			out_size = _scale[d_layers];

			weight = pa3(d_layers);
			neuron_value = pa2(layers);
			neuron_bias = pa2(d_layers);
			delta = pa2(layers);

			for (counter layer = 0; layer < d_layers; ++layer) // < d_layers for weight
			{
				counter neurons = scale[layer];
				counter nnl = scale[layer + 1];

				prec range = prec(std::sqrt(6.0 / (neurons + nnl))); // Xavier range
				std::uniform_real_distribution<prec> dist(-range, range);

				neuron_value[layer] = pa(neurons);
				neuron_bias[layer] = pa(neurons);
				delta[layer] = pa(neurons);

				weight[layer] = pa2(neurons);

				for (counter neuron = 0; neuron < neurons; ++neuron)
				{
					neuron_bias[layer][neuron] = (prec)dist(gen);
					weight[layer][neuron] = pa(nnl);

					for (counter neuron_nl = 0; neuron_nl < nnl; ++neuron_nl)
						weight[layer][neuron][neuron_nl] = (prec)dist(gen);
				}
			}

			neuron_value[d_layers] = pa(out_size);
			delta[d_layers] = pa(out_size);
			neuron_bias[0] = pa(0);
		}

		void save(std::string path)
		{
			std::ofstream w(path, std::ios::binary);
			if (w.is_open())
			{
				for (counter i = 0; i < layers; ++i) w << scale[i] << ';';
				w << '\n';

				size_t sv = sizeof(prec); // sv = size variable
				for (counter neuron = 0; neuron < inp_size; ++neuron)
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
					for (counter neuron = 0; neuron < inp_size; ++neuron)
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
#ifdef DEBUG
			if (neuron_value[0].size() != i.size())
			{
				throw std::runtime_error(std::string("ANNA: The input Tensor's size is not equal to the input layer Tensor's size.").c_str());
				return;
			}
#endif // DEBUG

			for (counter j = 0; j < inp_size; ++j) neuron_value[0][j] = i[j];

			for (counter l = 1; l < d_layers; ++l)
			{
				counter dl = l - 1;
				counter cn = scale[l];

				for (counter n = 0; n < cn; ++n)
				{
					counter mcln = scale[dl];
					neuron_value[l][n] = neuron_bias[l][n];

					for (counter cln = 0; cln < mcln; ++cln) // cln current last neuron
						neuron_value[l][n] += (neuron_value[dl][cln] * weight[dl][cln][n]);
					neuron_value[l][n] = sigmoid(neuron_value[l][n]);
				}
			}

			for (counter n = 0; n < out_size; ++n)
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
			for (counter n = 0; n < out_size; ++n)
			{
				prec ao = neuron_value[d_layers][n]; // caching the current neuron
				delta[d_layers][n] = (eo[n] - ao) * sigDeri(ao);
			}

			for (int64_t l = ddl; l >= 0; --l)
			{
				counter ml = scale[l];
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

			for (counter n = 0; n < out_size; ++n)
			{
				counter mddl = scale[ddl];
				for (counter nl = 0; nl < mddl; ++nl) // nl neuron last layer
					weight[ddl][nl][n] += neuron_value[ddl][nl] * delta[d_layers][n] * lr;
			}

			counter mn;
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
			counter md = d[0].size();
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

		counter getNParams()
		{
			counter rv = 0;
			for (counter l = 0; l < d_layers; ++l) rv += scale[l] * scale[l + 1] + scale[l];
			rv -= inp_size;
			rv -= out_size;
			return rv;
		}

		std::vector<counter> getScale() { return scale; }

		pa2 getBias() { return neuron_bias; }
		pa3 getWeight() { return weight; }

		void train(pa3& d, counter e)
		{
#ifdef DEBUG
			if ((d.size() != 2) || (d[0].size() != d[1].size())) throw std::runtime_error("ANNA: Error at train: Dataset d formatted wrong.");
#endif // DEBUG

			counter chunkSize = counter(std::ceil(d[0].size() / threads));
			counter chunkRemainder = d[0].size() % chunkSize;

			std::vector<ANNA> model(threads, *this);
			std::vector<std::thread> pool(threads);
			pa3 model_data_b(threads);
			std::vector<pa3> model_data_w(threads);

			for (counter t = 0; t < threads; ++t)
			{
				pool[t] = std::thread
				(
					[&]
					(counter id)
					{
						counter cs = chunkSize * id; // cs chunk start
						counter ce = cs + chunkSize; // ce chunk end

						for (counter i = 0; i < e; ++i)
						{
							for (counter s = cs; s < ce; ++s)
							{
								model[id].forward(d[0][s]);
								model[id].backward(d[1][s]);
							}
						}
					},
					t
				);
			}
			for (counter t = 0; t < threads; ++t)
			{
				if (pool[t].joinable()) pool[t].join();
				model_data_b[t] = model[t].getBias();
				model_data_w[t] = model[t].getWeight();
			}

			for (counter l = 0; l < d_layers; ++l)
			{
				counter mn = scale[l]; // mn max neuron
				for (counter n = 0; n < mn; ++n)
				{
					counter mnn = scale[l + 1]; // mnn max next neuron
					for (counter nn = 0; nn < mnn; ++nn)
					{
						prec x = prec(0.0f);
						for (counter t = 0; t < threads; ++t) x += model_data_w[t][l][n][nn];
						weight[l][n][nn] = x / threads;
					}
				}
			}

			for (counter l = 1; l < d_layers; ++l)
			{
				counter mn = scale[l];
				for (counter n = 0; n < mn; ++n)
				{
					prec x = prec(0.0f);
					for (counter t = 0; t < threads; ++t) x += model_data_b[t][l][n];
					neuron_bias[l][n] = x / threads;
				}
			}
		}
	};
};
