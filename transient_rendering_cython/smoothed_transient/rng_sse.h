/*
 * rng_sse.h
 *
 *  Created on: Nov 24, 2015
 *      Author: igkiou
 */

#ifndef RNG_SSE_H_
#define RNG_SSE_H_

#include <stdint.h>

#include "constants.h"

namespace rng {

typedef uint64_t SSEEngineSeedType;

class SSEEngine {
public:
	SSEEngine();

	template <typename IndexType>
	explicit SSEEngine(const IndexType seedValue);

	void seed(const SSEEngineSeedType seedValue = SSEEngineSeedType(5489));

	template <typename IndexType>
	inline void seed(const IndexType seedValue) {
		seed(static_cast<SSEEngineSeedType>(seedValue));
	}

	inline float operator()() {
		/* Trick from MTGP: generate an uniformly distributed
		   single precision number in [1,2) and subtract 1. */
		union {
			uint32_t u;
			float f;
		} x;
		x.u = ((nextULong() & 0xFFFFFFFF) >> 9) | 0x3f800000UL;
		return x.f - 1.0f;
	}

	~SSEEngine();

private:
	/// Return an integer on the [0, 2^63-1]-interval
	uint64_t nextULong();

	/// Return an integer on the [0, n)-interval
	uint32_t nextUInt(uint32_t n);

	/// Return an integer on the [0, n)-interval
	size_t nextSize(size_t n);

	/// Return a floating point value on the [0, 1) interval
	float nextFloat();

	struct State;
	State *mt;
};

} /* namespace rng */

#endif /* RNG_SSE_H_ */
