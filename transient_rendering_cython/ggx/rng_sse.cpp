/* 
   SIMD oriented Fast Mersenne Twister (SFMT) pseudorandom number generator
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/

   Copyright (c) 2006,2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima
   University. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials provided
         with the distribution.
       * Neither the name of the Hiroshima University nor the names of
         its contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   M. Saito and M. Matsumoto,
     ``SIMD-oriented Fast Mersenne Twister:
	   a 128-bit Pseudorandom Number Generator''
     Monte Carlo and Quasi-Monte Carlo Method 2006.
	 Springer (2008) 607--622.
	 DOI: 10.1007/978-3-540-74496-2_36
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.
*/

#include <limits>
#include <malloc.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_util.h"
#include "sse.h"

#include "rng_sse.h"

/************************ SFMT-19937 Parameters *******************************/

/* Mersenne Exponent. The period of the sequence 
 * is a multiple of 2^MEXP-1. */
#define MEXP 19937
/* SFMT generator has an internal state array of 128-bit integers,
 * and N is its size. */
#define N (MEXP / 128 + 1)
/* N32 is the size of internal state array when regarded as an array
 * of 32-bit integers.*/
#define N32 (N * 4)
/* N64 is the size of internal state array when regarded as an array
 * of 64-bit integers.*/
#define N64 (N * 2)

/* MEXP dependent values */
#define POS1	122
#define SL1	18
#define SL2	1
#define SR1	11
#define SR2	1
#define MSK1	0xdfffffefU
#define MSK2	0xddfecb7fU
#define MSK3	0xbffaffffU
#define MSK4	0xbffffff6U
#define PARITY1	0x00000001U
#define PARITY2	0x00000000U
#define PARITY3	0x00000000U
#define PARITY4	0x13c9e684U

/******************************************************************************/

// Helper elements
namespace rng {

namespace {

void * __restrict allocAligned(size_t size) {
	return memalign(L1_CACHE_LINE_SIZE, size);
}

void freeAligned(void *ptr) {
	free(ptr);
}

}

namespace {

/** Compute a hash value representing the SFMT parameters */
inline uint32_t sfmtHash() {
	// Based on boost::hash_combine
	const uint32_t data[] =
		{MEXP, POS1, SL1, SL2, SR1, SR2, MSK1, MSK2, MSK3, MSK4};
	uint32_t hash = 17;
	for (int i = 0; i < 10; ++i) 
		hash ^= data[i] + 0x9e3779b9U + (hash<<6) + (hash>>2);
	return hash;
}

/** 128-bit data structure */
union w128_t {
	__m128i si;
	uint32_t u[4];
};

/**
 * This function represents the recursion formula.
 * \param a a 128-bit part of the interal state array
 * \param b a 128-bit part of the interal state array
 * \param c a 128-bit part of the interal state array
 * \param d a 128-bit part of the interal state array
 * \param mask 128-bit mask
 * \return output
 */
//#if MTS_SFMT_SSE
inline __m128i mm_recursion(const  __m128i &a, const __m128i &b,
	const __m128i &c, const __m128i &d, const __m128i &mask) {
	__m128i v, x, y, z;

	x = _mm_load_si128(&a);
	y = _mm_srli_epi32(b, SR1);
	z = _mm_srli_si128(c, SR2);
	v = _mm_slli_epi32(d, SL1);
	z = _mm_xor_si128(z, x);
	z = _mm_xor_si128(z, v);
	x = _mm_slli_si128(x, SL2);
	y = _mm_and_si128(y, mask);
	z = _mm_xor_si128(z, x);
	z = _mm_xor_si128(z, y);
	return z;
}

} // namespace

// Actual state structure definition
struct SSEEngine::State {
	union {
		/** the 128-bit internal state array */
		w128_t sfmt[N];
		/** the 32bit integer version of the 128-bit internal state array */
		uint32_t psfmt32[N32];
		/** the 64bit integer version of the 128-bit internal state array */
		uint64_t psfmt64[N64];
	};

	/** index counter to the 32-bit internal state array */
	int idx;

	/** a parity check vector which certificate the period of 2^{MEXP} */
	const static uint32_t parity[4];

	/** Hash of the SFMT parameters */
	const static uint32_t s_magic;


	/* Default constructor, set the index to an invalid value */
	State() : idx(-1) {}

	/// Helper function to check whether the state is initialized
	inline bool isInitialized() const {
		return idx >= 0;
	}

	/**
	  * \brief This function initializes the internal state array with a 64-bit
	  * integer seed.
	  *
	  * \param seed a 64-bit integer used as the seed.
	  */
	void init_gen_rand(uint64_t seed);

	/**
	 * \brief This function initializes the internal state array,
	 * with an array of 32-bit integers used as the seeds
	 * \param init_key the array of 32-bit integers, used as a seed.
	 * \param key_length the length of init_key.
	 */
	void init_by_array(const uint32_t *init_key, int key_length);

	/**
	* This function generates and returns 64-bit pseudorandom number.
	* init_gen_rand or init_by_array must be called before this function.
	* The function gen_rand64 should not be called after gen_rand32,
	* unless an initialization is again executed. 
	* \return 64-bit pseudorandom number
	*/
	inline uint64_t gen_rand64() {
		if (idx >= N32) {
			gen_rand_all();
			idx = 0;
		}

		uint64_t r = psfmt64[idx / 2];
		idx += 2;
		return r;
	}

private:

	/**
	* This function represents a function used in the initialization
	* by init_by_array
	* \param x 32-bit integer
	* \return 32-bit integer
	*/
	static inline uint32_t func1(uint32_t x) {
		return (x ^ (x >> 27)) * (uint32_t)1664525UL;
	}

	/**
	* This function represents a function used in the initialization
	* by init_by_array
	* \param x 32-bit integer
	* \return 32-bit integer
	*/
	static inline uint32_t func2(uint32_t x) {
		return (x ^ (x >> 27)) * (uint32_t)1566083941UL;
	}

	/* This function certificate the period of 2^{MEXP} */
	void period_certification() {
		int inner = 0;
		int i, j;
		uint32_t work;

		for (i = 0; i < 4; ++i)
			inner ^= psfmt32[i] & parity[i];
		for (i = 16; i > 0; i >>= 1)
			inner ^= inner >> i;
		inner &= 1;
		/* check OK */
		if (inner == 1) {
			return;
		}
		/* check NG, and modification */
		for (i = 0; i < 4; ++i) {
			work = 1;
			for (j = 0; j < 32; ++j) {
				if ((work & parity[i]) != 0) {
					psfmt32[i] ^= work;
					return;
				}
				work = work << 1;
			}
		}
	}

	/*
	 * This function fills the internal state array with pseudorandom
	 * integers.
	 */
	inline void gen_rand_all() {
		int i;
		__m128i r, r1, r2, mask;
		mask = _mm_set_epi32(MSK4, MSK3, MSK2, MSK1);

		r1 = _mm_load_si128(&sfmt[N - 2].si);
		r2 = _mm_load_si128(&sfmt[N - 1].si);
		for (i = 0; i < N - POS1; ++i) {
			r = mm_recursion(sfmt[i].si, sfmt[i + POS1].si, r1, r2, mask);
			_mm_store_si128(&sfmt[i].si, r);
			r1 = r2;
			r2 = r;
		}
		for (; i < N; ++i) {
			r = mm_recursion(sfmt[i].si, sfmt[i + POS1 - N].si, r1, r2, mask);
			_mm_store_si128(&sfmt[i].si, r);
			r1 = r2;
			r2 = r;
		}
	}
};

const uint32_t SSEEngine::State::parity[4] = {PARITY1, PARITY2, PARITY3, PARITY4};
const uint32_t SSEEngine::State::s_magic = sfmtHash();


void SSEEngine::State::init_gen_rand(const uint64_t seed) {
	psfmt64[0] = seed;
	for (int i = 1; i < N64; ++i) {
		psfmt64[i] = (6364136223846793005ULL * (psfmt64[i-1] 
		              ^ (psfmt64[i-1] >> 62)) + i);
	}
	idx = N32;
	period_certification();
	Assert(isInitialized());
}

void SSEEngine::State::init_by_array(const uint32_t *init_key, int key_length) {
	int i, j, count;
	uint32_t r;
	int lag;
	int mid;
	int size = N * 4;

	if (size >= 623) {
		lag = 11;
	} else if (size >= 68) {
		lag = 7;
	} else if (size >= 39) {
		lag = 5;
	} else {
		lag = 3;
	}
	mid = (size - lag) / 2;

	memset(sfmt, 0x8b, sizeof(sfmt));
	if (key_length + 1 > N32) {
		count = key_length + 1;
	} else {
		count = N32;
	}
	r = func1(psfmt32[0] ^ psfmt32[mid]
	      ^ psfmt32[N32 - 1]);
	psfmt32[mid] += r;
	r += key_length;
	psfmt32[mid + lag] += r;
	psfmt32[0] = r;

	count--;
	for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
		r = func1(psfmt32[i] ^ psfmt32[(i + mid) % N32] 
		      ^ psfmt32[(i + N32 - 1) % N32]);
		psfmt32[(i + mid) % N32] += r;
		r += init_key[j] + i;
		psfmt32[(i + mid + lag) % N32] += r;
		psfmt32[i] = r;
		i = (i + 1) % N32;
	}
	for (; j < count; j++) {
		r = func1(psfmt32[i] ^ psfmt32[(i + mid) % N32] 
		      ^ psfmt32[(i + N32 - 1) % N32]);
		psfmt32[(i + mid) % N32] += r;
		r += i;
		psfmt32[(i + mid + lag) % N32] += r;
		psfmt32[i] = r;
		i = (i + 1) % N32;
	}
	for (j = 0; j < N32; j++) {
		r = func2(psfmt32[i] + psfmt32[(i + mid) % N32] 
		      + psfmt32[(i + N32 - 1) % N32]);
		psfmt32[(i + mid) % N32] ^= r;
		r -= i;
		psfmt32[(i + mid + lag) % N32] ^= r;
		psfmt32[i] = r;
		i = (i + 1) % N32;
	}

	idx = N32;
	period_certification();
	Assert(isInitialized());
}

SSEEngine::SSEEngine() : mt(NULL) {
	mt = (State *) allocAligned(sizeof(State));
	Assert(mt != NULL);
#if defined(_WIN32)
	seed();
#else
#if 0
	uint64_t buf[MT_N];
	memset(buf, 0, MT_N * sizeof(uint64_t)); /* Make GCC happy */
	ref<FileStream> urandom = new FileStream("/dev/urandom", FileStream::EReadOnly);
	urandom->readULongArray(buf, MT_N);
	seed(buf, MT_N);
#else
	seed();
#endif
#endif
}

template <typename IndexType>
SSEEngine::SSEEngine(const IndexType seedValue) : mt(NULL) {
	mt = (State *) allocAligned(sizeof(State));
	Assert(mt != NULL);
	seed(seedValue);
}

SSEEngine::~SSEEngine() {
	if (mt != NULL)
		freeAligned(mt);
}

void SSEEngine::seed(const SSEEngineSeedType s) {
	mt->init_gen_rand(s);
}

uint64_t SSEEngine::nextULong() {
	return mt->gen_rand64();
}


namespace {
	// Helper function to create bitmasks according to the size
	template <typename T> T makeBitmask(T n);
	
	template <> inline uint32_t makeBitmask(uint32_t n) {
		uint32_t bitmask = n;
		bitmask |= bitmask >> 1;
		bitmask |= bitmask >> 2;
		bitmask |= bitmask >> 4;
		bitmask |= bitmask >> 8;
		bitmask |= bitmask >> 16;
		return bitmask;
	}
	
	template <> inline uint64_t makeBitmask(uint64_t n) {
		uint64_t bitmask = n;
		bitmask |= bitmask >> 1;
		bitmask |= bitmask >> 2;
		bitmask |= bitmask >> 4;
		bitmask |= bitmask >> 8;
		bitmask |= bitmask >> 16;
		bitmask |= bitmask >> 32;
		return bitmask;
	}
	
	#if defined(MTS_AMBIGUOUS_SIZE_T)
	inline size_t makeBitmask(size_t n) {
		if (sizeof(size_t) == 8)
			return (size_t) makeBitmask((uint64_t) n);
		else
			return (size_t) makeBitmask((uint32_t) n);
	}
	#endif
} // namespace


uint32_t SSEEngine::nextUInt(uint32_t n) {
	/* Determine a bit mask */
	const uint32_t bitmask = makeBitmask(n);
	uint32_t result;

	/* Generate numbers until one in [0, n) is found */
	while ((result = (uint32_t) (nextULong() & bitmask)) >= n)
		;

	return result;
}

size_t SSEEngine::nextSize(size_t n) {
	/* Determine a bit mask */
	const size_t bitmask = makeBitmask(n);
	size_t result;

	/* Generate numbers until one in [0, n) is found */
	while ((result = (size_t) (nextULong() & bitmask)) >= n)
		;

	return result;
}

float SSEEngine::nextFloat() {
	return (*this)();
}

}	/* namespace rnd */
