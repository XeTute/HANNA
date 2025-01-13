#ifndef WHITEBOX_HPP
#define WHITEBOX_HPP

#include <algorithm>
#include <malloc.h>
#include <random>
#include <omp.h>

namespace wb
{
	using n = unsigned long long;

	template <typename _ty>
	class effarr
	{
	private:
		n size;
		_ty* mem;

		void dealloc()
		{
			if (mem)
			{
				std::free(mem);
				mem = nullptr;
				size = 0;
			}
		}

		bool alloc(n s)
		{
			dealloc();
			if (mem = (_ty*)std::malloc(s * sizeof(_ty)))
			{
				size = s;
				return true;
			}
			else
			{
				size = 0;
				return false;
			}
		}

	public:
		effarr() : size(0), mem(nullptr) {}
		effarr(n s) { alloc(s); }
		effarr(n s, const _ty& initwith)
		{
			alloc(s);
#pragma omp parallel for
			for (n e = 0; e < s; ++e)
				mem[e] = initwith;
		}

		bool resize(n s)
		{
			if (s == 0)
			{
				dealloc();
				return true;
			}
			if (s == size) return true;

			_ty* newaddr = (_ty*)std::malloc(s * sizeof(_ty));
			if (!newaddr) return false;

			if (mem && (size != 0)) std::copy(mem, mem + std::min(s, size), newaddr);
			mem = newaddr;
			size = s;

			return true;
		}

		void setX(_ty& x)
		{
#pragma omp parallel for
			for (n e = 0; e < size; ++e) mem[e] = x;
		}

		void random()
		{
			std::uniform_real_distribution<float> dist(-1.f, 1.f);
			std::mt19937 mt(std::random_device{}());
			for (n e = 0; e < size; ++e) mem[e] = dist(mt);
		}

		const _ty* data() { return mem; }

		void forceResetWithoutFree()
		{
			mem = nullptr;
			size = 0;
		}

		// Operators
		_ty& operator[] (const n& i) { return mem[i]; }

		void apply(void (*f) (_ty&, const _ty&), const effarr<_ty>& i)
		{
#pragma omp parallel for
			for (n e = 0; e < size; ++e) f(mem[e], i[e]);
		}

		void operator+= (effarr<_ty>& i) { apply([](_ty& a, const _ty& b) { a += b; }, i); }
		void operator-= (effarr<_ty>& i) { apply([](_ty& a, const _ty& b) { a -= b; }, i); }
		void operator*= (effarr<_ty>& i) { apply([](_ty& a, const _ty& b) { a *= b; }, i); }
		void operator/= (effarr<_ty>& i) { apply([](_ty& a, const _ty& b) { a /= b; }, i); }

		void operator= (const effarr<_ty>& i)
		{
			dealloc();
			size = i.size;
			std::copy(i.mem, i.mem + size, mem);
		}

		~effarr() { dealloc(); }
	};
}

#endif