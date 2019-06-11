/*
 * sampler.cpp
 *
 *  Created on: Nov 30, 2015
 *      Author: igkiou
 */

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "sampler.h"

namespace smp {

SamplerSet::SamplerSet(const int numSamplers)
			: m_numSamplers(numSamplers),
			  m_samplers() {
	m_samplers = new Sampler[m_numSamplers];
//	boost::mt19937 seedEngine(time(0));
	boost::mt19937 seedEngine;
	boost::uniform_int<unsigned int> seedDistr(0, UINT_MAX);
	boost::variate_generator<boost::mt19937&,
							boost::uniform_int<unsigned int> > seedGenerator(
													seedEngine, seedDistr);

	for (int iter = 0; iter < m_numSamplers; ++iter) {
		m_samplers[iter].seed(seedGenerator());
	}
}

SamplerSet::SamplerSet(const int numSamplers, const unsigned int seedValue)
		: m_numSamplers(numSamplers),
		  m_samplers() {
	m_samplers = new Sampler[m_numSamplers];
	for (int iter = 0; iter < m_numSamplers; ++iter) {
		m_samplers[iter].seed(seedValue);
	}
#ifndef NDEBUG
	std::cout << "seeding all samplers to " << seedValue << std::endl;
#endif
}

SamplerSet::SamplerSet(const std::vector<unsigned int>& seedVector)
		: m_numSamplers(seedVector.size()),
		  m_samplers() {
	m_samplers = new Sampler[m_numSamplers];
	for (int iter = 0; iter < m_numSamplers; ++iter) {
		m_samplers[iter].seed(seedVector[iter]);
	}
}

SamplerSet::~SamplerSet() {
	delete[] m_samplers;
}

}	/* namespace image */
