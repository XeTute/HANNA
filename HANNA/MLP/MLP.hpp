#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <valarray>
#include <vector>

#ifndef MLP_HPP
#define MLP_HPP

namespace MLP
{
	using n = std::size_t;
	using scales = std::vector<n>;
	using vec = std::valarray<float>;

	class MLP
	{
	private:

		// Parameters
		std::vector<vec> bias;
		std::vector<vec> neur;
		std::vector<vec> delta;
		std::vector<std::vector<vec>> weight;

		// Metadata
		scales nn; // number neurons
		n i;
		n l; // layer, layer = nn.size(), but std::vector::size() calculates each time, so consider this caching
		n dl; // decreased layer
		n ddl;

		void alloc() // after nn, l, dl, ddl, i is already init
		{
			bias.resize(dl);
			neur.resize(dl);
			weight.resize(dl);

			bias[0].resize(nn[0]);
			neur[0].resize(nn[0]);
			weight[0] = std::vector<vec>(nn[0], vec(i));

			for (n _l = 1; _l < dl; ++_l)
			{
				n nrns = nn[_l];
				bias[_l].resize(nrns);
				neur[_l].resize(nrns);
				weight[_l] = std::vector<vec>(nrns, vec(nn[_l - 1]));
			}
			rand();
		}

	public:

		float lr;

		MLP(): bias(0), neur(0), delta(0), weight(0), nn(0), i(0), l(0), dl(0), ddl(0), lr(0.f) {}
		MLP(scales scale) { birth(scale); }

		void birth(scales scale)
		{
			nn = scale;
			nn.erase(nn.begin());

			i = scale[0];
			l = scale.size();
			dl = l - 1;
			ddl = dl - 1;

			alloc();
		}

		void suicide()
		{
			bias.resize(0);
			neur.resize(0);
			weight.resize(0);
			nn.resize(0);
			l = 0;
			dl = 0;
			ddl = 0;
		}

		void rand() // xavier
		{
			std::mt19937 gen(std::random_device{}());
			for (std::size_t _l = 0; _l < dl; ++_l)
			{
				float range = std::sqrt(6.f / nn[_l] + (_l > 0 ? nn[_l - 1] : i));
				std::uniform_real_distribution<float> dist(-range, range);

				for (float& x : bias[_l]) x = dist(gen);
				for (vec& nrn : weight[_l])
					for (float& x : nrn) x = dist(gen);
			}
		}

		bool save(std::string path)
		{
			std::ofstream w(path, std::ios::out | std::ios::binary);
			if (!w || !w.good()) return false;

			if (!w.write((const char*)&l, sizeof(l))) return false;
			if (!w.write((const char*)&i, sizeof(i))) return false;
			if (!w.write((const char*)nn.data(), sizeof(nn[0]) * l)) return false;
			
			for (std::size_t _l = 0; _l < dl; ++_l)
			{
				if (!w.write((const char*)&bias[_l][0], sizeof(bias[0][0]) * bias[_l].size())) return false;
				std::size_t nrns = nn[_l];
				for (std::size_t nrn = 0; nrn < nrns; ++nrn)
					if (!w.write((const char*)&weight[_l][nrn][0], sizeof(weight[0][0][0]) * weight[_l][nrn].size()))
						return false;
			}

			return true;
		}

		bool load(std::string path)
		{
			std::ifstream r(path, std::ios::in | std::ios::binary);
			if (!r || !r.good()) return false;

			if (!r.read((char*)&l, sizeof(l))) return false;
			dl = l - 1;
			ddl = dl - 1;
			nn.resize(dl);

			if (!r.read((char*)&i, sizeof(i))) return false;
			if (!r.read((char*)nn.data(), sizeof(nn[0]) * l)) return false;
			alloc();

			for (std::size_t _l = 0; _l < dl; ++_l)
			{
				if (!r.read((char*)&bias[_l][0], sizeof(bias[0][0]) * bias[_l].size())) return false;
				std::size_t nrns = nn[_l];
				for (std::size_t nrn = 0; nrn < nrns; ++nrn)
					if (!r.read((char*)&weight[_l][nrn][0], sizeof(weight[0][0][0]) * weight[_l][nrn].size()))
						return false;
			}

			return true;
		}

		void enableTraining()
		{
			delta.resize(dl);
			for (std::size_t _l = 0; _l < dl; ++_l)
				delta[_l].resize(nn[_l]);
		}

		void disableTraining()
		{
			for (std::size_t _l = 0; _l < dl; ++_l)
				delta[_l].resize(0);
			delta.resize(0);
		}

		void forward(const vec& input, float (*activation) (const float&))
		{
			std::size_t nrns = nn[0];
			for (std::size_t nrn = 0; nrn < nrns; ++nrn)
			{
				neur[0][nrn] = (weight[0][nrn] * input).sum() + bias[0][nrn];
				neur[0][nrn] = activation(neur[0][nrn]);
			}

			for (std::size_t _l = 1; _l < dl; ++_l)
			{
				nrns = nn[_l];
				for (std::size_t nrn = 0; nrn < nrns; ++nrn)
				{
					neur[_l][nrn] = (weight[_l][nrn] * neur[_l - 1]).sum() + bias[_l][nrn];
					neur[_l][nrn] = activation(neur[_l][nrn]);
				}
			}
		}
		const vec& out() { return neur[ddl]; }
		const vec& out(const vec& input, float (*activation) (const float&)) { forward(input, activation); return out(); }

		void gradDesc(const vec& last_input, const vec& output, float (*activationDV) (const float&))
		{
			std::size_t nrns = nn[ddl];
			float cache;

			delta[ddl] = (neur[ddl] - output) * neur[ddl].apply(activationDV);

			// Calculate errors & cache them in delta
			for (std::intmax_t _l = ddl - 1; _l >= 0; --_l) // signed for this one number underflow =(
			{
				std::size_t il = _l + 1; // il increased layer
				nrns = nn[_l];
				std::size_t nxt_nrns = nn[il];

				for (std::size_t nrn = 0; nrn < nrns; ++nrn)
				{
					float cache = 0.f;
					for (std::size_t nxt_nrn = 0; nxt_nrn < nxt_nrns; ++nxt_nrn)
						cache += weight[il][nxt_nrn][nrn] * delta[il][nxt_nrn];
					delta[_l][nrn] = cache * activationDV(neur[_l][nrn]);
				}
			}

			// Actually apply these calculated values
			nrns = nn[0];
			for (std::size_t nrn = 0; nrn < nrns; ++nrn)
			{
				cache = lr * delta[0][nrn];
				bias[0][nrn] -= cache;
				weight[0][nrn] -= cache * last_input;
			}

			for (std::size_t _l = 1; _l < dl; ++_l)
			{
				nrns = nn[_l];
				for (std::size_t nrn = 0; nrn < nrns; ++nrn)
				{
					cache = lr * delta[_l][nrn];
					bias[_l][nrn] -= cache;
					weight[_l][nrn] -= cache * neur[_l - 1];
				}
			}
		}

		void train
		(
			const std::vector<vec>& input, const std::vector<vec>& output,
			float (*activation) (const float&), float (*activationDV) (const float&),
			std::size_t epochs
		)
		{
			std::size_t samples = std::min(input.size(), output.size());
			for (std::size_t e = 0; e < epochs; ++e)
			{
				for (std::size_t s = 0; s < samples; ++s)
				{
					forward(input[s], activation);
					gradDesc(input[s], output[s], activationDV);
				}
			}
		}

		~MLP() { suicide(); }
	};
}

#endif