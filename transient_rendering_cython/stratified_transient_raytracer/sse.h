/*
 * sse.h
 *
 *  Created on: Nov 24, 2015
 *      Author: igkiou
 */

/*
 * Inspired by mitsuba.
 */
#ifndef SSE_H_
#define SSE_H_

#include <stdio.h>
#include <stdint.h>

#define MM_ALIGN16             __attribute__ ((aligned (16)))
#define STACK_ALIGN16(t)       reinterpret_cast<float *>((reinterpret_cast<size_t>(t)+0xF) & ~(size_t) 0xF)

/* ========= SSE intrinsics ========= */
/* Include SSE intrinsics header file */
#include <emmintrin.h>

#define splat_ps(ps, i)          _mm_shuffle_ps   ((ps),(ps), (i<<6) | (i<<4) | (i<<2) | i)
#define splat_epi32(ps, i)       _mm_shuffle_epi32((ps), (i<<6) | (i<<4) | (i<<2) | i)
#define mux_ps(sel, op1, op2)    _mm_or_ps   (_mm_and_ps   ((sel), (op1)), _mm_andnot_ps   ((sel), (op2)))
#define mux_epi32(sel, op1, op2) _mm_or_si128(_mm_and_si128((sel), (op1)), _mm_andnot_si128((sel), (op2)))
#define enable_fpexcept_sse()	 _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO))
#define query_fpexcept_sse()	 (~_MM_GET_EXCEPTION_MASK() & (_MM_MASK_INVALID | _MM_MASK_DIV_ZERO))
#define disable_fpexcept_sse()	 _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() | (_MM_MASK_INVALID | _MM_MASK_DIV_ZERO))
#define load1_epi32(i)           _mm_shuffle_epi32(_mm_cvtsi32_si128(i), 0)
#define negate_ps(val)           _mm_xor_ps((val), SSEConstants::negation_mask.ps)

#define pstoepi32(ps)            _mm_castps_si128(ps)
#define epi32tops(pi)            _mm_castsi128_ps(pi)

union SSEVector {
	__m128 ps;
	__m128i pi;
	float f[4];
	int32_t	i[4];
	uint32_t ui[4];
	struct { float   f0,f1,f2,f3; };
	struct { int32_t i0,i1,i2,i3; };
	struct { uint32_t ui0,ui1,ui2,ui3; };

	inline SSEVector() {
	}

	explicit SSEVector(__m128 ps)
		: ps(ps) {
	}

	explicit SSEVector(float f0, float f1, float f2, float f3)
		: f0(f0), f1(f1), f2(f2), f3(f3) {
	}

	explicit SSEVector(float f) : f0(f), f1(f), f2(f), f3(f) {}

	explicit SSEVector(int32_t i0, int32_t i1, int32_t i2, int32_t i3)
		: i0(i0), i1(i1), i2(i2), i3(i3) {
	}

	explicit SSEVector(int32_t i) : i0(i), i1(i), i2(i), i3(i) {}

	explicit SSEVector(uint32_t ui0, uint32_t ui1, uint32_t ui2, uint32_t ui3)
		: ui0(ui0), ui1(ui1), ui2(ui2), ui3(ui3) {
	}

	explicit SSEVector(uint32_t ui) : ui0(ui), ui1(ui), ui2(ui), ui3(ui) {}

	inline SSEVector &operator=(const SSEVector &vec) {
		ps = vec.ps;
		return *this;
	}
};

/**
 * \brief Some useful constant values for use with SSE
 * \headerfile mitsuba/core/sse.h mitsuba/mitsuba.h
 */
class SSEConstants {
public:
	/// (0, 0, 0, 0)
	static const MM_ALIGN16 SSEVector zero;
	/// (1, 1, 1, 1)
	static const MM_ALIGN16 SSEVector one;
	/// (flt_max, flt_max, flt_max, flt_max)
	static const MM_ALIGN16 SSEVector max;
	/// (eps, eps, eps, eps)
	static const MM_ALIGN16 SSEVector eps;
	/// (1+eps, 1+eps, 1+eps, 1+eps)
	static const MM_ALIGN16 SSEVector op_eps;
	/// (1-eps, 1-eps, 1-eps, 1-eps)
	static const MM_ALIGN16 SSEVector om_eps;
	/// (+inf, +inf, +inf, +inf)
	static const MM_ALIGN16 SSEVector p_inf;
	/// (-inf, -inf, -inf, -inf)
	static const MM_ALIGN16 SSEVector n_inf;
	/// (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
	static const MM_ALIGN16 SSEVector ffffffff;
	/// (0x80000000, 0x80000000, 0x80000000, 0x80000000)
	static const MM_ALIGN16 SSEVector negation_mask;
};

#endif /* SSE_ */
