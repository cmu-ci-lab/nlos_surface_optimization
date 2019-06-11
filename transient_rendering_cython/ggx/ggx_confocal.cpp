/*
 * ggx2.h
 *
 *  Created on: Nov 8, 2018
 *      Author: igkiou
 *	Based on Mitsuba.
 */

#include "ggx_confocal.h"
#include <common/math/math.h>	


float eval(const float m_alpha, const Vector& normal, const Vector& w) {
	/* Stop if this component was not requested */
	if (dot(normal, w) <= 0) {
		return float(0.0f);
	}
	/* Evaluate the microfacet normal distribution */
	const float Dval = D(m_alpha, normal, w);
	if (Dval == 0) {
		return float(0.0f);
	}
	/* Smith's shadow-masking function */
	const float Gval = G(m_alpha, normal, w);
	/* Calculate the total amount of reflection */
	return Dval * Gval / 4.0f;
}
	
float D(const float m_alpha, const Vector &normal, const Vector &w) {

	float nw = dot(normal, w);
	if (nw <= 0) {
		return 0.0f;
	}

	float nw2 = nw * nw;
	float beckmannExponent = (1.0f - nw2) / (m_alpha * m_alpha) / nw2;

	/* GGX / Trowbridge-Reitz distribution function for rough surfaces */
	float root = (1.0f + beckmannExponent) * nw2;
	float result = 1.0f / (M_PI * m_alpha * m_alpha * root * root);

	/* Prevent potential numerical issues in other stages of the model */
	if (result * nw < 1e-20f) {
		result = 0;
	}

	return result;
}
	
float G(const float m_alpha, const Vector &normal, const Vector &w) {
	float G1val = G1(m_alpha, normal, w);
	return G1val * G1val;
}

float G1(const float m_alpha, const Vector &normal, const Vector &w) {
	/* Ensure consistent orientation (can't see the back
	   of the microfacet from the front and vice versa) */
	float nw = dot(normal, w);
	if (nw <= 0) {
		return 0.0f;
	}

	if ((nw >= 1.0f) || (nw <= -1.0f)) {
		return 1.0f;
	}

	float root = m_alpha * m_alpha + (1.0f - m_alpha * m_alpha) * nw * nw;
	return 2.0f / (nw + embree::sqrt(root));
}



float eval_adiff(const float m_alpha, const Vector& normal, const Vector& w) {

	/* Stop if this component was not requested */
	if (dot(normal, w) <= 0) {
		return float(0.0f);
	}

	/* Evaluate the microfacet normal distribution */
	const float Dval = D(m_alpha, normal, w);
	if (Dval == 0) {
		return float(0.0f);
	}

	/* Smith's shadow-masking function */
	const float Gval = G(m_alpha, normal, w);

	/* Compute D prime */
	const float Dprime = D_adiff(m_alpha, normal, w);

	/* Compute G_prime */
	const float Gprime = G_adiff(m_alpha, normal, w);

	/* Calculate the total amount of reflection */
	return (Dprime * Gval + Gprime * Dval) / 4.0f;
}

float D_adiff(const float m_alpha, const Vector &normal, const Vector &w) {
	float nw = dot(normal, w);
	if (nw <= 0) {
		return 0.0f;
	}

	float nw2 = nw * nw;
	float a2 = m_alpha * m_alpha;
	float val = a2 * nw2 - nw2 + 1;

	// Plugged in from matlab.
	return -(2.0f * m_alpha * (a2 * nw2 + nw2 - 1)) / (M_PI * val * val * val);

}

float G_adiff(const float m_alpha, const Vector &normal, const Vector &w) {
	return 2.0f * G1_adiff(m_alpha, normal, w) * G1(m_alpha, normal, w);
}

float G1_adiff(const float m_alpha, const Vector &normal, const Vector &w) {
	float nw = dot(normal, w);
	if (nw <= 0) {
		return 0.0f;
	}

	/* Perpendicular incidence -- no shadowing/masking */
	/* TODO: Confirm derivative here. For now, 0*/
	if ((nw >= 1.0f) || (nw <= -1.0f)) {
		return 0.0f;
	}

	float nw2 = nw * nw;
	float val = embree::sqrt(m_alpha * m_alpha - nw2 * (m_alpha * m_alpha - 1));
	float root = nw + val;
	// Plugged in from matlab.
	return 2.0f * m_alpha * (nw2 - 1.0f) / (val * root * root);
}

void eval_nwdiff(const float m_alpha, const Vector& normal, const Vector& w, Vector& dnormal, Vector& dw) {

	/* Stop if this component was not requested */
	if (dot(normal, w) <= 0) {
		return;
	}

	/* Evaluate the microfacet normal distribution */
	const float Dval = D(m_alpha, normal, w);
	if (Dval == 0) {
		return;
	}

	/* Smith's shadow-masking function */
	const float Gval = G(m_alpha, normal, w);

	/* Compute normal differential */

	/* Compute dG/dn */
	const float GprimeDn = G_ndiff(m_alpha, normal, w);

	/* Compute dD/dn */
	const float DprimeDn = D_ndiff(m_alpha, normal, w);

	const float Dscale = (DprimeDn * Gval + GprimeDn * Dval) / 4.0f;

	dnormal = Dscale * w;
	dw = Dscale * normal;
}

float eval_nwsdiff(const float m_alpha, const Vector& normal, const Vector& w) {

	/* Stop if this component was not requested */
	if (dot(normal, w) <= 0) {
		return float(0.0f);
	}

	/* Evaluate the microfacet normal distribution */
	const float Dval = D(m_alpha, normal, w);
	if (Dval == 0) {
		return float(0.0f);
	}

	/* Smith's shadow-masking function */
	const float Gval = G(m_alpha, normal, w);

	/* Compute normal differential */

	/* Compute dG/dn */
	const float GprimeDn = G_ndiff(m_alpha, normal, w);

	/* Compute dD/dn */
	const float DprimeDn = D_ndiff(m_alpha, normal, w);

	return (DprimeDn * Gval + GprimeDn * Dval) / 4.0f;
}

float D_ndiff(const float m_alpha, const Vector &normal, const Vector &w) {
	float nw = dot(normal, w);
	if (nw <= 0) {
		return float(0.0f);
	}

	float nw2 = nw * nw;
	float a2 = m_alpha * m_alpha;
	float root = (a2 - 1.0f) * nw2 + 1.0f;

	// Plugged in from matlab.
	return -(4.0f * a2 * nw * (a2 - 1.0f)) / (M_PI * root * root * root);
}

float G_ndiff(const float m_alpha, const Vector &normal, const Vector &w) {
	return 2.0f * G1_ndiff(m_alpha, normal, w) * G1(m_alpha, normal, w);
}

float G1_ndiff(const float m_alpha, const Vector &normal, const Vector &w) {
	float nw = dot(normal, w);
	if (nw <= 0) {
		return 0.0f;
	}

	/* Perpendicular incidence -- no shadowing/masking */
	/* TODO: Confirm derivative here. For now, 0*/
	if ((nw >= 1.0f) || (nw <= -1.0f)) {
		return 0.0f;
	}

	float nw2 = nw * nw;
	float a2 = m_alpha * m_alpha;
	float temp = embree::sqrt(a2 - nw2 * (a2 - 1.0f));
	float root = nw + temp;

	return -2.0f * (1.0f - (nw * (a2 - 1.0f)) / temp) / root / root;
}

