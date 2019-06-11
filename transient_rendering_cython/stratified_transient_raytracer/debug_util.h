/*
 * debug_util.h
 *
 *  Created on: Sep 4, 2012
 *      Author: igkiou
 */

#ifndef DEBUG_UTIL_H_
#define DEBUG_UTIL_H_

#include <stdio.h>
#include <stdarg.h>

// Inspired by mitsuba-0.4.1
#ifdef NDEBUG
#define Assert(cond) ((void) 0)
#define AssertEx(cond, explanation) ((void) 0)
#else
/* Assertions */
// Assert that a condition is true
#define Assert(cond) do { \
		if (!(cond)) fprintf(stderr, "Assertion \"%s\" failed in %s:%i\n", \
		#cond, __FILE__, __LINE__); \
	} while (0)

// Assertion with a customizable error explanation
#define AssertEx(cond, explanation) do { \
		if (!(cond)) fprintf(stderr, "Assertion \"%s\" failed in %s:%i (" explanation ")\n", \
		#cond, __FILE__, __LINE__); \
	} while (0)
#endif

#endif /* DEBUG_UTIL_H_ */
