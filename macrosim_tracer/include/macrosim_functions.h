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
#include <vector_types.h>
#include <vector_functions.h>
#include <math.h>

// __forceinline__ works in cuda, VS, and with gcc.  Leave it as macro in case
// we need to make this per-platform or we want to switch off inlining globally.
#ifndef OPTIXU_INLINE 
#  define OPTIXU_INLINE_DEFINED 1
#  define OPTIXU_INLINE __forceinline__
#endif // OPTIXU_INLINE 

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
/* Some types didn't exist until CUDA 3.0.  CUDA_VERSION isn't defined while
* building CUDA code, so we also need to check the CUDART_VERSION value. */
//#if (CUDA_VERSION < 3000) //|| (CUDART_VERSION < 3000)
#if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < 4000)
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

//static __inline__ __host__ __device__ double dot(const double3& a, const double3& b)
//{
//  return a.x * b.x + a.y * b.y + a.z * b.z;
//}







/* double2 functions */
/******************************************************************************/

/** comparison
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(double2 a, double2 b)
{
	return ( (a.x == b.x) && (a.y == b.y) );
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(double2 a, double2 b)
{
	return ( (a.x != b.x) && (a.y != b.y) );
}
/** @} */  

/* double3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(double3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
OPTIXU_INLINE RT_HOSTDEVICE double3 make_double3(double s)
{
    return make_double3(s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 make_double3(double2 a)
{
    return make_double3(a.x, a.y, 0.0f);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 make_double3(double2 a, double s)
{
    return make_double3(a.x, a.y, s);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 make_double3(int3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}
OPTIXU_INLINE RT_HOSTDEVICE double3 make_double3(float3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

OPTIXU_INLINE RT_HOSTDEVICE double2 make_double2(uint2 a)
{
    return make_double2(double(a.x), double(a.y));
}

OPTIXU_INLINE RT_HOSTDEVICE double3 make_double3(uint3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

OPTIXU_INLINE RT_HOSTDEVICE double4 make_double4(uint4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}

OPTIXU_INLINE RT_HOSTDEVICE double3x3 make_rotMatrix(double3 rot)
{
	double3x3 t;
	t=make_double3x3(1,0,0, 0,1,0, 0,0,1);
	return t;
}
/** @} */

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator-(const double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}
/** @} */

///* min */
//static __inline__ RT_HOSTDEVICE double3 dmind(double3 a, double3 b)
//{
//	//return make_double3(dmind(a.x,b.x), dmind(a.y,b.y), dmind(a.z,b.z));
//	double ax=a.x;
//	double bx=b.x;
//	double atest=dmind((double)1, (double)0);
//	return make_double3(b.x, b.y, b.z);
//}
//
///* max */
//static __inline__ RT_HOSTDEVICE double3 dmaxd(double3 a, double3 b)
//{
//	return make_double3(dmaxd(a.x,b.x), dmaxd(a.y,b.y), dmaxd(a.z,b.z));
//}

/** addition
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator+(double a, double3 b)
{
    return make_double3(a + b.x, a + b.y, a + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(double3 &a, double3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** comparison
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const double3 &a, const double3 &b)
{
	return ( (a.x == b.x) && (a.y == b.y) && (a.z == b.z) );
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const double3 &a,const  double3 &b)
{
	return ( (a.x != b.x) && (a.y != b.y) && (a.z != b.z) );
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator-(double a, double3 b)
{
    return make_double3(a - b.x, a - b.y, a - b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(double3 &a, double3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator*(double3 a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator*(double s, double3 a)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(double3 &a, double3 s)
{
    a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(double3 &a, double s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE double3 operator/(double3 a, long3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE double3 operator/(long3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE double3 operator/(double3 a, double s)
{
    double inv = (double)1.0 / s;
    return a * inv;
}
OPTIXU_INLINE RT_HOSTDEVICE double3 operator/(double s, double3 a)
{
    return make_double3( s/a.x, s/a.y, s/a.z );
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(double3 &a, double s)
{
    double inv = (double)1.0 / s;
    a *= inv;
}
/** @} */

/* copy sign-bit from src value to dst value */
inline RT_HOSTDEVICE double copy_sign(double dst, double src)
{
	//union {
 //   double d;
 //   unsigned long i;
 // } v1, v2, v3;
 // v1.d = src;
 // v2.d = dst;
 // v3.i = (v2.i & 0x7fffffffffffffff) | (v1.i & 0x8000000000000000);
 // //v3.i = (v2.i & 0x7fffffff) | (v1.i & 0x80000000);
 // return v3.d;
	// theres gotta be a nice way to do it without branching...
	if (src < 0) return -dst;
	return dst;
}

/** lerp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 lerp(double3 a, double3 b, double t)
{
    return a + t*(b-a);
}

OPTIXU_INLINE RT_HOSTDEVICE double lerp(double a, double b, double t)
{
    return a + t*(b-a);
}
/** @} */

/* clamp */
//OPTIXU_INLINE RT_HOSTDEVICE double3 clamp(double3 v, double a, double b)
//{
//    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
//}
//
//OPTIXU_INLINE RT_HOSTDEVICE double3 clamp(double3 v, double3 a, double3 b)
//{
//    return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
//}

/** dot product
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
/** @} */

/** cross product
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
/** @} */

/** length
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double length(double3 v)
{
    return sqrt(dot(v, v));
}
/** @} */

/** normalize
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 normalize(double3 v)
{
    double invLen = (double)1.0 / sqrt(dot(v, v));
    return v * invLen;
}
/** @} */

/** floor
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 floor(const double3 v)
{
	return make_double3(::floor(v.x), ::floor(v.y), ::floor(v.z));
}
/** @} */

/** reflect
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 reflect(double3 i, double3 n)
{
	return i - (double)2.0 * n * dot(n,i);
}
/** @} */

/** faceforward
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 faceforward(double3 n, double3 i, double3 nref)
{
	return n * copy_sign( (double)1.0, dot(i, nref) );
}
/** @} */



/* double4x4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double4x4 make_double4x4(double4 s1, double4 s2, double4 s3, double4 s4)
{
    return make_double4x4(s1.x, s2.x, s3.x, s4.x, s1.y, s2.y, s3.y, s4.y, s1.z, s2.z, s3.z, s4.z, s1.w, s2.w, s3.w, s4.w);
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator*(double4x4 m, double3 a)
{
	double3 b;
	b.x = m.m11*a.x+m.m12*a.y+m.m13*a.z+m.m14*1;
	b.y = m.m21*a.x+m.m22*a.y+m.m23*a.z+m.m24*1;
	b.z = m.m31*a.x+m.m32*a.y+m.m33*a.z+m.m34*1;
	return b;
}

OPTIXU_INLINE RT_HOSTDEVICE double4x4 operator*(double4x4 m1, double4x4 m2)
{
	double4x4 mOut;
	mOut.m11 = m1.m11*m2.m11+m1.m12*m2.m21+m1.m13*m2.m31+m1.m14*m2.m41;
	mOut.m12 = m1.m11*m2.m12+m1.m12*m2.m22+m1.m13*m2.m32+m1.m14*m2.m42;
	mOut.m13 = m1.m11*m2.m13+m1.m12*m2.m23+m1.m13*m2.m33+m1.m14*m2.m43;
	mOut.m14 = m1.m11*m2.m14+m1.m12*m2.m24+m1.m13*m2.m34+m1.m14*m2.m44;
	mOut.m21 = m1.m21*m2.m11+m1.m22*m2.m21+m1.m23*m2.m31+m1.m24*m2.m41;
	mOut.m22 = m1.m21*m2.m12+m1.m22*m2.m22+m1.m23*m2.m32+m1.m24*m2.m42;
	mOut.m23 = m1.m21*m2.m13+m1.m22*m2.m23+m1.m23*m2.m33+m1.m24*m2.m43;
	mOut.m24 = m1.m21*m2.m14+m1.m22*m2.m24+m1.m23*m2.m34+m1.m24*m2.m44;
	mOut.m31 = m1.m31*m2.m11+m1.m32*m2.m21+m1.m33*m2.m31+m1.m34*m2.m41;
	mOut.m32 = m1.m31*m2.m12+m1.m32*m2.m22+m1.m33*m2.m32+m1.m34*m2.m42;
	mOut.m33 = m1.m31*m2.m13+m1.m32*m2.m23+m1.m33*m2.m33+m1.m34*m2.m43;
	mOut.m34 = m1.m31*m2.m14+m1.m32*m2.m24+m1.m33*m2.m34+m1.m34*m2.m44;
	mOut.m41 = m1.m41*m2.m11+m1.m42*m2.m21+m1.m43*m2.m31+m1.m44*m2.m41;
	mOut.m42 = m1.m41*m2.m12+m1.m42*m2.m22+m1.m43*m2.m32+m1.m44*m2.m42;
	mOut.m43 = m1.m41*m2.m13+m1.m42*m2.m23+m1.m43*m2.m33+m1.m44*m2.m43;
	mOut.m44 = m1.m41*m2.m14+m1.m42*m2.m24+m1.m43*m2.m34+m1.m44*m2.m44;
	return mOut;
}
/** @} */

/* double3x3 functions */
/******************************************************************************/

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3x3 make_double3x3(double3 s1, double3 s2, double3 s3)
{
    return make_double3x3(s1.x,s2.x,s3.x, 
                        s1.y,s2.y,s3.y, 
                        s1.z,s2.z,s3.z);
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3 operator*(double3x3 m, double3 a)
{
	double3 b;
	b.x = m.m11*a.x+m.m12*a.y+m.m13*a.z;
	b.y = m.m21*a.x+m.m22*a.y+m.m23*a.z;
	b.z = m.m31*a.x+m.m32*a.y+m.m33*a.z;
	return b;
}

OPTIXU_INLINE RT_HOSTDEVICE double3x3 operator*(double3x3 m1, double3x3 m2)
{
	double3x3 mOut;
	mOut.m11 = m1.m11*m2.m11+m1.m12*m2.m21+m1.m13*m2.m31;
	mOut.m12 = m1.m11*m2.m12+m1.m12*m2.m22+m1.m13*m2.m32;
	mOut.m13 = m1.m11*m2.m13+m1.m12*m2.m23+m1.m13*m2.m33;
	mOut.m21 = m1.m21*m2.m11+m1.m22*m2.m21+m1.m23*m2.m31;
	mOut.m22 = m1.m21*m2.m12+m1.m22*m2.m22+m1.m23*m2.m32;
	mOut.m23 = m1.m21*m2.m13+m1.m22*m2.m23+m1.m23*m2.m33;
	mOut.m31 = m1.m31*m2.m11+m1.m32*m2.m21+m1.m33*m2.m31;
	mOut.m32 = m1.m31*m2.m12+m1.m32*m2.m22+m1.m33*m2.m32;
	mOut.m33 = m1.m31*m2.m13+m1.m32*m2.m23+m1.m33*m2.m33;
	return mOut;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double3x3 operator/(double3x3 m1, double s)
{
	return make_double3x3(m1.m11/s,m1.m12/s,m1.m13/s, 
						  m1.m21/s,m1.m22/s,m1.m23/s,
						  m1.m31/s,m1.m32/s,m1.m33/s);
}
/** @} */

/** invert
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE double det(double3x3 mat)
{
	return mat.m11*mat.m22*mat.m33+mat.m12*mat.m23*mat.m31+mat.m13*mat.m21*mat.m32-mat.m13*mat.m22*mat.m31-mat.m12*mat.m21*mat.m33-mat.m11*mat.m23*mat.m32;
}

OPTIXU_INLINE RT_HOSTDEVICE double3x3 inv(double3x3 mat)
{
	double3x3 l_mat;
	l_mat=make_double3x3( mat.m22*mat.m33-mat.m23*mat.m32,mat.m13*mat.m32-mat.m12*mat.m33,mat.m12*mat.m23-mat.m13*mat.m22, 
						mat.m23*mat.m31-mat.m21*mat.m33,mat.m11*mat.m33-mat.m13*mat.m31,mat.m13*mat.m21-mat.m11*mat.m23,
						mat.m21*mat.m32-mat.m22*mat.m31,mat.m12*mat.m31-mat.m11*mat.m32,mat.m11*mat.m22-mat.m12*mat.m21);
	return l_mat/det(mat);
}
/** @} */


#ifdef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE
#endif


#endif /* !__MACROSIM_FUNCTIONS_H__ */
