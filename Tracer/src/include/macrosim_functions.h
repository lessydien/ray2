/***********************************************************************
 This file is part of ITO-MacroSim.

    ITO-MacroSim is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITO-MacroSim is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
************************************************************************/

#if !defined(__MACROSIM_FUNCTIONS_H__)
#define __MACROSIM_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "host_defines.h"
#include "macrosim_types.h"
#include <nppversion.h>

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/



#if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < 4200)

 static __inline__ __host__ __device__ double1 make_double1(double x)
 {
   double1 t; t.x = x; return t;
 }

 static __inline__ __host__ __device__ double2 make_double2(double x, double y)
 {
   double2 t; t.x = x; t.y = y; return t;
 }

 static __inline__ __host__ __device__ double3 make_double3(double x, double y, double z)
 {
   double3 t; t.x = x; t.y = y; t.z = z; return t;
 }

  static __inline__ __host__ __device__ double4 make_double4(double x, double y, double z, double w)
 {
   double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
 }

#endif

static __inline__ __host__ __device__ double4x4 make_double4x4(double s11, double s12, double s13, double s14, double s21, double s22, double s23, double s24, double s31, double s32, double s33, double s34, double s41, double s42, double s43, double s44)
{
    double4x4 t; 
	t.m11 = s11; t.m12 = s12; t.m13 = s13; t.m14 = s14;
	t.m21 = s21; t.m22 = s22; t.m23 = s23; t.m24 = s24;
	t.m31 = s31; t.m32 = s32; t.m33 = s33; t.m34 = s34;
	t.m41 = s41; t.m42 = s42; t.m43 = s43; t.m44 = s44;
	return t;
}

static __inline__ __host__ __device__ double3x3 make_double3x3(double s11, double s12, double s13, double s21, double s22, double s23, double s31, double s32, double s33)
{
    double3x3 t; 
	t.m11 = s11; t.m12 = s12; t.m13 = s13;
	t.m21 = s21; t.m22 = s22; t.m23 = s23;
	t.m31 = s31; t.m32 = s32; t.m33 = s33;
	return t;
}




#endif /* !__MACROSIM_FUNCTIONS_H__ */
