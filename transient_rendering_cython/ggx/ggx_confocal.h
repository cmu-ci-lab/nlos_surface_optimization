/*
 * ggx2.h
 *
 *  Created on: Nov 8, 2018
 *      Author: igkiou
 *	Based on Mitsuba.
 */

#ifndef GGX_H_
#define GGX_H_
#include <common/math/vec.h>

using namespace embree;
typedef Vec3fa Vector;
	
float eval(const float m_alpha, const Vector & normal, const Vector & w);
float D(const float m_alpha, const Vector &normal, const Vector &w);
float G(const float m_alpha, const Vector &normal, const Vector &w);
float G1(const float m_alpha, const Vector &normal, const Vector &w);
float eval_adiff(const float m_alpha, const Vector& normal, const Vector& w);
float D_adiff(const float m_alpha, const Vector &normal, const Vector &w);
float G_adiff(const float m_alpha, const Vector &normal, const Vector &w);
float G1_adiff(const float m_alpha, const Vector &normal, const Vector &w);
void eval_nwdiff(const float m_alpha, const Vector& normal, const Vector& w, Vector& dnormal, Vector& dw);
float eval_nwsdiff(const float m_alpha, const Vector& normal, const Vector& w);
float D_ndiff(const float m_alpha, const Vector &normal, const Vector &w);
float G_ndiff(const float m_alpha, const Vector &normal, const Vector &w);
float G1_ndiff(const float m_alpha, const Vector &normal, const Vector &w);

#endif /* GGX_H_ */
