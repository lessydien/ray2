
/*
 * Copyright (c) 1993 - 2010 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
    This file implements common mathematical operations on vector types
    (float3, float4 etc.) since these are not provided as standard by CUDA.

    The syntax is modelled on the Cg standard library.

    This file has also been modified from the original cutil_math.h file.
    cutil_math.h is a subset of this file, and you should use this file in place
    of any cutil_math.h file you wish to use.
*/

#ifndef __optixu_optixu_math_namespace_h__
#define __optixu_optixu_math_namespace_h__

#include <optix.h>                     // For RT_HOSTDEVICE
#include <internal/optix_datatypes.h>  // For optix::Ray
#include "optixu_vector_functions.h"
#include <optix_sizet.h>

#if !defined(_WIN32)
// On posix systems uint and ushort are defined when including this file, so we need to
// guarantee this file gets included in order to get these typedefs.
#  include <sys/types.h>
#endif

// #define these constants such that we are sure
// 32b floats are emitted in ptx
#ifndef M_Ef
#define M_Ef        2.71828182845904523536f
#endif
#ifndef M_LOG2Ef
#define M_LOG2Ef    1.44269504088896340736f
#endif
#ifndef M_LOG10Ef
#define M_LOG10Ef   0.434294481903251827651f
#endif
#ifndef M_LN2f
#define M_LN2f      0.693147180559945309417f
#endif
#ifndef M_LN10f
#define M_LN10f     2.30258509299404568402f
#endif
#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f     0.785398163397448309616f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif
#ifndef M_2_PIf
#define M_2_PIf     0.636619772367581343076f
#endif
#ifndef M_2_SQRTPIf
#define M_2_SQRTPIf 1.12837916709551257390f
#endif
#ifndef M_SQRT2f
#define M_SQRT2f    1.41421356237309504880f
#endif
#ifndef M_SQRT1_2f
#define M_SQRT1_2f  0.707106781186547524401f
#endif

/******************************************************************************/
namespace optix {
#if defined(_WIN32) && !defined(RT_UINT_USHORT_DEFINED)
  // uint and ushort are not already defined on Windows systems or they could have been
  // defined in optixu_math.h.
  typedef unsigned int uint;
  typedef unsigned short ushort;
#else
  // On Posix systems these typedefs are defined in the global namespace, and to avoid
  // conflicts, we'll pull them into this namespace for consistency.
  using ::uint;
  using ::ushort;
#endif //defined(_WIN32)
} // end namespace optix

#if !defined(__CUDACC__)
/* Functions that CUDA provides for device code but are lacking on some host platform */

#include <math.h>


// We need to declare these functions to define them in Windows and to override the system
// library version in Posix systems (it's declared extern).  On Posix the system versions
// are not inlined and cause slower perforance.  In addition on non-Windows systems we
// can't declare them in a namespace, because we need to override the one declared extern
// in the global namespace and subsequent overloaded versions of need to qualify their
// call with the global namespace to avoid auto-casting from float to float3 and friends.

#if defined(_WIN32)
namespace optix {
#endif

inline float fminf(float a, float b)
{
  return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
  return a > b ? a : b;
}

/* copy sign-bit from src value to dst value */
inline float copysignf(float dst, float src)
{
  union {
    float f;
    unsigned int i;
  } v1, v2, v3;
  v1.f = src;
  v2.f = dst;
  v3.i = (v2.i & 0x7fffffff) | (v1.i & 0x80000000);

  return v3.f;
}

#if defined(_WIN32)
} // end namespace optix
#endif

#endif // #ifndef __CUDACC__

namespace optix {
  // On Posix systems these functions are defined in the global namespace, but we need to
  // pull them into the optix namespace in order for them to be on the same level as
  // the other overloaded functions in optix::.

#if !defined(_WIN32) || defined (__CUDACC__)
  // These functions are in the global namespace on POSIX (not _WIN32) and in CUDA C.
  using ::fminf;
  using ::fmaxf;
  using ::copysignf;
//  using ::copy_sign;
#endif
  using ::expf;
  using ::floorf;

  // These are defined by CUDA in the global namespace.
#ifdef __CUDACC__
  using ::min;
  using ::max;
#else
#if defined(_WIN32) && !defined(NOMINMAX)
#  error "optixu_math_namespace.h needs NOMINMAX defined on windows."
#endif
  inline int max(int a, int b)
  {
    return a > b ? a : b;
  }

  inline int min(int a, int b)
  {
    return a < b ? a : b;
  }
#endif

} // end namespace optix


namespace optix {

/* Bit preserving casting functions */
/******************************************************************************/

#ifdef __CUDACC__

  using ::float_as_int;
  using ::int_as_float;

#else

inline int float_as_int( float f )
{
  union {
    float f;
    int i;
  } v1;

  v1.f = f;
  return v1.i;
}


inline float int_as_float( int i )
{
  union {
    float f;
    int i;
  } v1;

  v1.i = i;
  return v1.f;
}

#endif 


/* float functions */
/******************************************************************************/

/* lerp */
inline RT_HOSTDEVICE float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

/* bilerp */
inline RT_HOSTDEVICE float bilerp(float x00, float x10, float x01, float x11,
                                  float u, float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/* clamp */
inline RT_HOSTDEVICE float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

/* int2 functions */
/******************************************************************************/

/* negate */
inline RT_HOSTDEVICE int2 operator-(int2 a)
{
    return make_int2(-a.x, -a.y);
}

/* addition */
inline RT_HOSTDEVICE int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline RT_HOSTDEVICE void operator+=(int2 &a, int2 b)
{
    a.x += b.x; a.y += b.y;
}

/* subtract */
inline RT_HOSTDEVICE int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline RT_HOSTDEVICE int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline RT_HOSTDEVICE void operator-=(int2 &a, int2 b)
{
    a.x -= b.x; a.y -= b.y;
}

/* multiply */
inline RT_HOSTDEVICE int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline RT_HOSTDEVICE int2 operator*(int2 a, int s)
{
    return make_int2(a.x * s, a.y * s);
}
inline RT_HOSTDEVICE int2 operator*(int s, int2 a)
{
    return make_int2(a.x * s, a.y * s);
}
inline RT_HOSTDEVICE void operator*=(int2 &a, int s)
{
    a.x *= s; a.y *= s;
}

/* float2 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline RT_HOSTDEVICE float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

/* negate */
inline RT_HOSTDEVICE float2 operator-(float2 a)
{
    return make_float2(-a.x, -a.y);
}

/* addition */
inline RT_HOSTDEVICE float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline RT_HOSTDEVICE float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline RT_HOSTDEVICE float2 operator+(float a, float2 b)
{
    return make_float2(a + b.x, a + b.y);
}
inline RT_HOSTDEVICE void operator+=(float2 &a, float2 b)
{
    a.x += b.x; a.y += b.y;
}

/* subtract */
inline RT_HOSTDEVICE float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline RT_HOSTDEVICE float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline RT_HOSTDEVICE float2 operator-(float a, float2 b)
{
    return make_float2(a - b.x, a - b.y);
}
inline RT_HOSTDEVICE void operator-=(float2 &a, float2 b)
{
    a.x -= b.x; a.y -= b.y;
}

/* multiply */
inline RT_HOSTDEVICE float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline RT_HOSTDEVICE float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
inline RT_HOSTDEVICE float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
inline RT_HOSTDEVICE void operator*=(float2 &a, float2 s)
{
    a.x *= s.x; a.y *= s.y;
}
inline RT_HOSTDEVICE void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

/* divide */
inline RT_HOSTDEVICE float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline RT_HOSTDEVICE float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline RT_HOSTDEVICE float2 operator/(float s, float2 a)
{
    return make_float2( s/a.x, s/a.y );
}
inline RT_HOSTDEVICE void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

/* lerp */
inline RT_HOSTDEVICE float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}

/* bilerp */
inline RT_HOSTDEVICE float2 bilerp(float2 x00, float2 x10, float2 x01, float2 x11,
                                   float u, float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/* clamp */
inline RT_HOSTDEVICE float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline RT_HOSTDEVICE float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

/* dot product */
inline RT_HOSTDEVICE float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

/* length */
inline RT_HOSTDEVICE float length(float2 v)
{
    return sqrtf(dot(v, v));
}

/* normalize */
inline RT_HOSTDEVICE float2 normalize(float2 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/* floor */
inline RT_HOSTDEVICE float2 floor(const float2 v)
{
  return make_float2(::floorf(v.x), ::floorf(v.y));
}

/* reflect */
inline RT_HOSTDEVICE float2 reflect(float2 i, float2 n)
{
	return i - 2.0f * n * dot(n,i);
}

/* faceforward */
inline RT_HOSTDEVICE float2 faceforward(float2 n, float2 i, float2 nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/* exp */
inline RT_HOSTDEVICE float2 expf(float2 v)
{
  return make_float2(::expf(v.x), ::expf(v.y));
}


/* float3 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline RT_HOSTDEVICE float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline RT_HOSTDEVICE float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline RT_HOSTDEVICE float3 make_float3(float4 a)
{
   return make_float3(a.x, a.y, a.z);  /* discards w */
}
inline RT_HOSTDEVICE float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

/* negate */
inline RT_HOSTDEVICE float3 operator-(float3 a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

/* min */
static __inline__ RT_HOSTDEVICE float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

/* max */
static __inline__ RT_HOSTDEVICE float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

/* addition */
inline RT_HOSTDEVICE float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline RT_HOSTDEVICE float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline RT_HOSTDEVICE float3 operator+(float a, float3 b)
{
    return make_float3(a + b.x, a + b.y, a + b.z);
}
inline RT_HOSTDEVICE void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

/* subtract */
inline RT_HOSTDEVICE float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline RT_HOSTDEVICE float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline RT_HOSTDEVICE float3 operator-(float a, float3 b)
{
    return make_float3(a - b.x, a - b.y, a - b.z);
}
inline RT_HOSTDEVICE void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

/* multiply */
inline RT_HOSTDEVICE float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline RT_HOSTDEVICE float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline RT_HOSTDEVICE float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline RT_HOSTDEVICE void operator*=(float3 &a, float3 s)
{
    a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
inline RT_HOSTDEVICE void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

/* divide */
inline RT_HOSTDEVICE float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline RT_HOSTDEVICE float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline RT_HOSTDEVICE float3 operator/(float s, float3 a)
{
    return make_float3( s/a.x, s/a.y, s/a.z );
}
inline RT_HOSTDEVICE void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

/* lerp */
inline RT_HOSTDEVICE float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

/* bilerp */
inline RT_HOSTDEVICE float3 bilerp(float3 x00, float3 x10, float3 x01, float3 x11,
                                   float u, float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/* clamp */
inline RT_HOSTDEVICE float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline RT_HOSTDEVICE float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

/* dot product */
inline RT_HOSTDEVICE float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* cross product */
inline RT_HOSTDEVICE float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/* length */
inline RT_HOSTDEVICE float length(float3 v)
{
    return sqrtf(dot(v, v));
}

/* normalize */
inline RT_HOSTDEVICE float3 normalize(float3 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/* floor */
inline RT_HOSTDEVICE float3 floor(const float3 v)
{
  return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/* reflect */
inline RT_HOSTDEVICE float3 reflect(float3 i, float3 n)
{
	return i - 2.0f * n * dot(n,i);
}

/* faceforward */
inline RT_HOSTDEVICE float3 faceforward(float3 n, float3 i, float3 nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/* exp */
inline RT_HOSTDEVICE float3 expf(float3 v)
{
  return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z));
}


/* float4 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline RT_HOSTDEVICE float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline RT_HOSTDEVICE float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline RT_HOSTDEVICE float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

/* negate */
inline RT_HOSTDEVICE float4 operator-(float4 a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

/* min */
static __inline__ RT_HOSTDEVICE float4 fminf(float4 a, float4 b)
{
	return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

/* max */
static __inline__ RT_HOSTDEVICE float4 fmaxf(float4 a, float4 b)
{
	return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

/* addition */
inline RT_HOSTDEVICE float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline RT_HOSTDEVICE float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline RT_HOSTDEVICE float4 operator+(float a, float4 b)
{
    return make_float4(a + b.x, a + b.y, a + b.z,  a + b.w);
}
inline RT_HOSTDEVICE void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

/* subtract */
inline RT_HOSTDEVICE float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline RT_HOSTDEVICE float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline RT_HOSTDEVICE float4 operator-(float a, float4 b)
{
    return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}
inline RT_HOSTDEVICE void operator-=(float4 &a, float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

/* multiply */
inline RT_HOSTDEVICE float4 operator*(float4 a, float4 s)
{
    return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
inline RT_HOSTDEVICE float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline RT_HOSTDEVICE float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline RT_HOSTDEVICE void operator*=(float4 &a, float4 s)
{
    a.x *= s.x; a.y *= s.y; a.z *= s.z; a.w *= s.w;
}
inline RT_HOSTDEVICE void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

/* divide */
inline RT_HOSTDEVICE float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline RT_HOSTDEVICE float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline RT_HOSTDEVICE float4 operator/(float s, float4 a)
{
    return make_float4( s/a.x, s/a.y, s/a.z, s/a.w );
}
inline RT_HOSTDEVICE void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

/* lerp */
inline RT_HOSTDEVICE float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

/* bilerp */
inline RT_HOSTDEVICE float4 bilerp(float4 x00, float4 x10, float4 x01, float4 x11,
                                   float u, float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/* clamp */
inline RT_HOSTDEVICE float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline RT_HOSTDEVICE float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

/* dot product */
inline RT_HOSTDEVICE float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* length */
inline RT_HOSTDEVICE float length(float4 r)
{
    return sqrtf(dot(r, r));
}

/* normalize */
inline RT_HOSTDEVICE float4 normalize(float4 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/* floor */
inline RT_HOSTDEVICE float4 floor(const float4 v)
{
  return make_float4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z), ::floorf(v.w));
}

/* reflect */
inline RT_HOSTDEVICE float4 reflect(float4 i, float4 n)
{
	return i - 2.0f * n * dot(n,i);
}

/* faceforward */
inline RT_HOSTDEVICE float4 faceforward(float4 n, float4 i, float4 nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/* exp */
inline RT_HOSTDEVICE float4 expf(float4 v)
{
  return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/* double2 functions */
/******************************************************************************/

/* comparison */
__inline RT_HOSTDEVICE bool operator==(double2 a, double2 b)
{
	return ( (a.x == b.x) && (a.y == b.y) );
}

__inline RT_HOSTDEVICE bool operator!=(double2 a, double2 b)
{
	return ( (a.x != b.x) && (a.y != b.y) );
}

/* double3 functions */
/******************************************************************************/

__inline RT_HOSTDEVICE float3 make_float3(double3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

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
	if (src > 0) return dst;
	if (src < 0) return -dst;
	return dst;
}

/* additional constructors */
__inline RT_HOSTDEVICE double3 make_double3(double s)
{
    return make_double3(s, s, s);
}
__inline RT_HOSTDEVICE double3 make_double3(double2 a)
{
    return make_double3(a.x, a.y, 0.0f);
}
__inline RT_HOSTDEVICE double3 make_double3(double2 a, double s)
{
    return make_double3(a.x, a.y, s);
}
__inline RT_HOSTDEVICE double3 make_double3(int3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}
__inline RT_HOSTDEVICE double3 make_double3(float3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

__inline RT_HOSTDEVICE double2 make_double2(uint2 a)
{
    return make_double2(double(a.x), double(a.y));
}

__inline RT_HOSTDEVICE double3 make_double3(uint3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

__inline RT_HOSTDEVICE double4 make_double4(uint4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}

__inline RT_HOSTDEVICE double3x3 make_rotMatrix(double3 rot)
{
	double3x3 t;
	t=make_double3x3(1,0,0, 0,1,0, 0,0,1);
	return t;
}

/* double4x4 functions */
/******************************************************************************/

/* additional constructors */
__inline RT_HOSTDEVICE double4x4 make_double4x4(double4 s1, double4 s2, double4 s3, double4 s4)
{
    return make_double4x4(s1.x, s2.x, s3.x, s4.x, s1.y, s2.y, s3.y, s4.y, s1.z, s2.z, s3.z, s4.z, s1.w, s2.w, s3.w, s4.w);
}

__inline RT_HOSTDEVICE double3 operator*(double4x4 m, double3 a)
{
	double3 b;
	b.x = m.m11*a.x+m.m12*a.y+m.m13*a.z+m.m14*1;
	b.y = m.m21*a.x+m.m22*a.y+m.m23*a.z+m.m24*1;
	b.z = m.m31*a.x+m.m32*a.y+m.m33*a.z+m.m34*1;
	return b;
}

__inline RT_HOSTDEVICE double4x4 operator*(double4x4 m1, double4x4 m2)
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

/* double3x3 functions */
/******************************************************************************/

/* additional constructors */
__inline RT_HOSTDEVICE double3x3 make_double3x3(double3 s1, double3 s2, double3 s3)
{
    return make_double3x3(s1.x,s2.x,s3.x, s1.y,s2.y,s3.y, s1.z,s2.z,s3.z);
}

__inline RT_HOSTDEVICE double3 operator*(double3x3 m, double3 a)
{
	double3 b;
	b.x = m.m11*a.x+m.m12*a.y+m.m13*a.z;
	b.y = m.m21*a.x+m.m22*a.y+m.m23*a.z;
	b.z = m.m31*a.x+m.m32*a.y+m.m33*a.z;
	return b;
}

__inline RT_HOSTDEVICE double3x3 operator/(double3x3 m1, double s)
{
	return make_double3x3(m1.m11/s,m1.m12/s,m1.m13/s, 
						  m1.m21/s,m1.m22/s,m1.m23/s,
						  m1.m31/s,m1.m32/s,m1.m33/s);
}

__inline RT_HOSTDEVICE double det(double3x3 mat)
{
	return mat.m11*mat.m22*mat.m33+mat.m12*mat.m23*mat.m31+mat.m13*mat.m21*mat.m32-mat.m13*mat.m22*mat.m31-mat.m12*mat.m21*mat.m33-mat.m11*mat.m23*mat.m32;
}

__inline RT_HOSTDEVICE double3x3 inv(double3x3 mat)
{
	double3x3 l_mat;
	l_mat=make_double3x3( mat.m22*mat.m33-mat.m23*mat.m32,mat.m13*mat.m32-mat.m12*mat.m33,mat.m12*mat.m23-mat.m13*mat.m22, 
						mat.m23*mat.m31-mat.m21*mat.m33,mat.m11*mat.m33-mat.m13*mat.m31,mat.m13*mat.m21-mat.m11*mat.m23,
						mat.m21*mat.m32-mat.m22*mat.m31,mat.m12*mat.m31-mat.m11*mat.m32,mat.m11*mat.m22-mat.m12*mat.m21);
	return l_mat/det(mat);
}

__inline RT_HOSTDEVICE double3x3 operator*(double3x3 m1, double3x3 m2)
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

/* negate */
__inline RT_HOSTDEVICE double3 operator-(const double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

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

/* addition */
__inline RT_HOSTDEVICE double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__inline RT_HOSTDEVICE double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
__inline RT_HOSTDEVICE double3 operator+(double a, double3 b)
{
    return make_double3(a + b.x, a + b.y, a + b.z);
}
__inline RT_HOSTDEVICE void operator+=(double3 &a, double3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

/* comparison */
__inline RT_HOSTDEVICE bool operator==(double3 a, double3 b)
{
	return ( (a.x == b.x) && (a.y == b.y) && (a.z == b.z) );
}

__inline RT_HOSTDEVICE bool operator!=(double3 a, double3 b)
{
	return ( (a.x != b.x) && (a.y != b.y) && (a.z != b.z) );
}

/* subtract */
__inline RT_HOSTDEVICE double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__inline RT_HOSTDEVICE double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
__inline RT_HOSTDEVICE double3 operator-(double a, double3 b)
{
    return make_double3(a - b.x, a - b.y, a - b.z);
}
__inline RT_HOSTDEVICE void operator-=(double3 &a, double3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

/* multiply */
__inline RT_HOSTDEVICE double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__inline RT_HOSTDEVICE double3 operator*(double3 a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}
__inline RT_HOSTDEVICE double3 operator*(double s, double3 a)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}
__inline RT_HOSTDEVICE void operator*=(double3 &a, double3 s)
{
    a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
__inline RT_HOSTDEVICE void operator*=(double3 &a, double s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

/* divide */
__inline RT_HOSTDEVICE double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__inline RT_HOSTDEVICE double3 operator/(double3 a, long3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__inline RT_HOSTDEVICE double3 operator/(long3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__inline RT_HOSTDEVICE double3 operator/(double3 a, double s)
{
    double inv = (double)1.0 / s;
    return a * inv;
}
__inline RT_HOSTDEVICE double3 operator/(double s, double3 a)
{
    return make_double3( s/a.x, s/a.y, s/a.z );
}
__inline RT_HOSTDEVICE void operator/=(double3 &a, double s)
{
    double inv = (double)1.0 / s;
    a *= inv;
}

/* lerp */
__inline RT_HOSTDEVICE double3 lerp(double3 a, double3 b, double t)
{
    return a + t*(b-a);
}

/* clamp */
//__inline RT_HOSTDEVICE double3 clamp(double3 v, double a, double b)
//{
//    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
//}
//
//__inline RT_HOSTDEVICE double3 clamp(double3 v, double3 a, double3 b)
//{
//    return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
//}

/* dot product */
__inline RT_HOSTDEVICE double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* cross product */
__inline RT_HOSTDEVICE double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/* length */
__inline RT_HOSTDEVICE double length(double3 v)
{
    return sqrt(dot(v, v));
}

/* normalize */
__inline RT_HOSTDEVICE double3 normalize(double3 v)
{
    double invLen = (double)1.0 / sqrt(dot(v, v));
    return v * invLen;
}

/* floor */
__inline RT_HOSTDEVICE double3 floor(const double3 v)
{
	return make_double3(::floor(v.x), ::floor(v.y), ::floor(v.z));
}

/* reflect */
__inline RT_HOSTDEVICE double3 reflect(double3 i, double3 n)
{
	return i - (double)2.0 * n * dot(n,i);
}

/* faceforward */
__inline RT_HOSTDEVICE double3 faceforward(double3 n, double3 i, double3 nref)
{
	return n * copy_sign( (double)1.0, dot(i, nref) );
}

/* lerp */
__inline RT_HOSTDEVICE double lerp(double a, double b, double t)
{
    return a + t*(b-a);
}

/* exp */
//__inline RT_HOSTDEVICE double3 expf(double3 v)
//{
//  return make_double3(expf(v.x), expf(v.y), expf(v.z));
//}

/* clamp */
//__inline RT_HOSTDEVICE double clamp(double f, double a, double b)
//{
//    return dmaxd(a, dmind(f, b));
//}
/* complex_t3 functions */
/*****************************************************************************/




/* int3 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline RT_HOSTDEVICE int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

/* negate */
inline RT_HOSTDEVICE int3 operator-(int3 a)
{
    return make_int3(-a.x, -a.y, -a.z);
}

/* min */
inline RT_HOSTDEVICE int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/* max */
inline RT_HOSTDEVICE int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/* addition */
inline RT_HOSTDEVICE int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline RT_HOSTDEVICE void operator+=(int3 &a, int3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

/* subtract */
inline RT_HOSTDEVICE int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline RT_HOSTDEVICE void operator-=(int3 &a, int3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

/* multiply */
inline RT_HOSTDEVICE int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline RT_HOSTDEVICE int3 operator*(int3 a, int s)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
inline RT_HOSTDEVICE int3 operator*(int s, int3 a)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
inline RT_HOSTDEVICE void operator*=(int3 &a, int s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

/* divide */
inline RT_HOSTDEVICE int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline RT_HOSTDEVICE int3 operator/(int3 a, int s)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}
inline RT_HOSTDEVICE int3 operator/(int s, int3 a)
{
    return make_int3(s /a.x, s / a.y, s / a.z);
}
inline RT_HOSTDEVICE void operator/=(int3 &a, int s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

/* clamp */
inline RT_HOSTDEVICE int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

inline RT_HOSTDEVICE int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline RT_HOSTDEVICE int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}


/* uint3 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE uint3 make_uint3(unsigned int s)
{
    return make_uint3(s, s, s);
}
inline RT_HOSTDEVICE uint3 make_uint3(float3 a)
{
    return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}

/* min */
inline RT_HOSTDEVICE uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/* max */
inline RT_HOSTDEVICE uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/* addition */
inline RT_HOSTDEVICE uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline RT_HOSTDEVICE void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

/* subtract */
inline RT_HOSTDEVICE uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline RT_HOSTDEVICE void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

/* multiply */
inline RT_HOSTDEVICE uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline RT_HOSTDEVICE uint3 operator*(uint3 a, unsigned int s)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline RT_HOSTDEVICE uint3 operator*(unsigned int s, uint3 a)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline RT_HOSTDEVICE void operator*=(uint3 &a, unsigned int s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

/* divide */
inline RT_HOSTDEVICE uint3 operator/(uint3 a, uint3 b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline RT_HOSTDEVICE uint3 operator/(uint3 a, unsigned int s)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline RT_HOSTDEVICE uint3 operator/(unsigned int s, uint3 a)
{
    return make_uint3(s / a.x, s / a.y, s / a.z);
}
inline RT_HOSTDEVICE void operator/=(uint3 &a, unsigned int s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

/* clamp */
inline RT_HOSTDEVICE unsigned int clamp(unsigned int f, unsigned int a, unsigned int b)
{
    return max(a, min(f, b));
}

inline RT_HOSTDEVICE uint3 clamp(uint3 v, unsigned int a, unsigned int b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline RT_HOSTDEVICE uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

/* int4 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}

/* equality */
inline RT_HOSTDEVICE bool operator==(int4 a, int4 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

/* uint4 functions */
/******************************************************************************/

/* additional constructors */
inline RT_HOSTDEVICE uint4 make_uint4(unsigned int s)
{
    return make_uint4(s, s, s, s);
}




/* Narrowing */
inline RT_HOSTDEVICE int2 make_int2(int3 v0) { return make_int2( v0.x, v0.y ); }
inline RT_HOSTDEVICE int2 make_int2(int4 v0) { return make_int2( v0.x, v0.y ); }
inline RT_HOSTDEVICE int3 make_int3(int4 v0) { return make_int3( v0.x, v0.y, v0.z ); }
inline RT_HOSTDEVICE uint2 make_uint2(uint3 v0) { return make_uint2( v0.x, v0.y ); }
inline RT_HOSTDEVICE uint2 make_uint2(uint4 v0) { return make_uint2( v0.x, v0.y ); }
inline RT_HOSTDEVICE uint3 make_uint3(uint4 v0) { return make_uint3( v0.x, v0.y, v0.z ); }
inline RT_HOSTDEVICE float2 make_float2(float3 v0) { return make_float2( v0.x, v0.y ); }
inline RT_HOSTDEVICE float2 make_float2(float4 v0) { return make_float2( v0.x, v0.y ); }

__inline RT_HOSTDEVICE double2 make_double2(double3 v0) { return make_double2( v0.x, v0.y ); }
__inline RT_HOSTDEVICE double2 make_double2(double4 v0) { return make_double2( v0.x, v0.y ); }


/* Assemble from smaller vectors */
inline RT_HOSTDEVICE int3 make_int3(int v0, int2 v1) { return make_int3( v0, v1.x, v1.y ); }
inline RT_HOSTDEVICE int3 make_int3(int2 v0, int v1) { return make_int3( v0.x, v0.y, v1 ); }
inline RT_HOSTDEVICE int4 make_int4(int v0, int v1, int2 v2) { return make_int4( v0, v1, v2.x, v2.y ); }
inline RT_HOSTDEVICE int4 make_int4(int v0, int2 v1, int v2) { return make_int4( v0, v1.x, v1.y, v2 ); }
inline RT_HOSTDEVICE int4 make_int4(int2 v0, int v1, int v2) { return make_int4( v0.x, v0.y, v1, v2 ); }
inline RT_HOSTDEVICE int4 make_int4(int v0, int3 v1) { return make_int4( v0, v1.x, v1.y, v1.z ); }
inline RT_HOSTDEVICE int4 make_int4(int3 v0, int v1) { return make_int4( v0.x, v0.y, v0.z, v1 ); }
inline RT_HOSTDEVICE int4 make_int4(int2 v0, int2 v1) { return make_int4( v0.x, v0.y, v1.x, v1.y ); }
inline RT_HOSTDEVICE uint3 make_uint3(unsigned int v0, uint2 v1) { return make_uint3( v0, v1.x, v1.y ); }
inline RT_HOSTDEVICE uint3 make_uint3(uint2 v0, unsigned int v1) { return make_uint3( v0.x, v0.y, v1 ); }
inline RT_HOSTDEVICE uint4 make_uint4(unsigned int v0, unsigned int v1, uint2 v2) { return make_uint4( v0, v1, v2.x, v2.y ); }
inline RT_HOSTDEVICE uint4 make_uint4(unsigned int v0, uint2 v1, unsigned int v2) { return make_uint4( v0, v1.x, v1.y, v2 ); }
inline RT_HOSTDEVICE uint4 make_uint4(uint2 v0, unsigned int v1, unsigned int v2) { return make_uint4( v0.x, v0.y, v1, v2 ); }
inline RT_HOSTDEVICE uint4 make_uint4(unsigned int v0, uint3 v1) { return make_uint4( v0, v1.x, v1.y, v1.z ); }
inline RT_HOSTDEVICE uint4 make_uint4(uint3 v0, unsigned int v1) { return make_uint4( v0.x, v0.y, v0.z, v1 ); }
inline RT_HOSTDEVICE uint4 make_uint4(uint2 v0, uint2 v1) { return make_uint4( v0.x, v0.y, v1.x, v1.y ); }
inline RT_HOSTDEVICE float3 make_float3(float v0, float2 v1) { return make_float3( v0, v1.x, v1.y ); }
inline RT_HOSTDEVICE float4 make_float4(float v0, float v1, float2 v2) { return make_float4( v0, v1, v2.x, v2.y ); }
inline RT_HOSTDEVICE float4 make_float4(float v0, float2 v1, float v2) { return make_float4( v0, v1.x, v1.y, v2 ); }
inline RT_HOSTDEVICE float4 make_float4(float2 v0, float v1, float v2) { return make_float4( v0.x, v0.y, v1, v2 ); }
inline RT_HOSTDEVICE float4 make_float4(float v0, float3 v1) { return make_float4( v0, v1.x, v1.y, v1.z ); }
inline RT_HOSTDEVICE float4 make_float4(float2 v0, float2 v1) { return make_float4( v0.x, v0.y, v1.x, v1.y ); }

__inline RT_HOSTDEVICE double3 make_double3(double v0, double2 v1) { return make_double3( v0, v1.x, v1.y ); }
__inline RT_HOSTDEVICE double4 make_double4(double v0, double v1, double2 v2) { return make_double4( v0, v1, v2.x, v2.y ); }
__inline RT_HOSTDEVICE double4 make_double4(double v0, double2 v1, double v2) { return make_double4( v0, v1.x, v1.y, v2 ); }
__inline RT_HOSTDEVICE double4 make_double4(double2 v0, double v1, double v2) { return make_double4( v0.x, v0.y, v1, v2 ); }
__inline RT_HOSTDEVICE double4 make_double4(double v0, double3 v1) { return make_double4( v0, v1.x, v1.y, v1.z ); }
__inline RT_HOSTDEVICE double4 make_double4(double2 v0, double2 v1) { return make_double4( v0.x, v0.y, v1.x, v1.y ); }

static __inline__ RT_HOSTDEVICE float fmaxf(float2 a)
{
  return fmaxf(a.x, a.y);
}

static __inline__ RT_HOSTDEVICE float fmaxf(float3 a)
{
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

static __inline__ RT_HOSTDEVICE float fmaxf(float4 a)
{
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}

static __inline__ RT_HOSTDEVICE float fminf(float2 a)
{
  return fminf(a.x, a.y);
}

static __inline__ RT_HOSTDEVICE float fminf(float3 a)
{
  return fminf(fminf(a.x, a.y), a.z);
}

static __inline__ RT_HOSTDEVICE float fminf(float4 a)
{
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}

inline RT_HOSTDEVICE float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline RT_HOSTDEVICE float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline RT_HOSTDEVICE float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

/* Common helper functions */
/******************************************************************************/

/* Return a smooth value in [0,1], where the transition from 0
   to 1 takes place for values of x in [edge0,edge1]. */
inline RT_HOSTDEVICE float smoothstep( float edge0, float edge1, float x )
{
    /*assert( edge1 > edge0 ); */
    const float t = clamp( (x-edge0) / (edge1-edge0), 0.0f, 1.0f );
    return t*t * ( 3.0f - 2.0f*t );
}

/* Simple mapping from [0,1] to a temperature-like RGB color. */
inline RT_HOSTDEVICE float3 temperature( float t )
{
    const float b = t < 0.25f ? smoothstep( -0.25f, 0.25f, t ) : 1.0f-smoothstep( 0.25f, 0.5f, t );
    const float g = t < 0.5f  ? smoothstep( 0.0f, 0.5f, t ) :
               (t < 0.75f ? 1.0f : 1.0f-smoothstep( 0.75f, 1.0f, t ));
    const float r = smoothstep( 0.5f, 0.75f, t );
    return make_float3( r, g, b );
}


/**
 *  \brief Intersect ray with CCW wound triangle
 */ 
__device__ __inline__ bool intersect_triangle( const Ray& ray,
                                               const float3& p0,
                                               const float3& p1,
                                               const float3& p2,
                                               float3& n,
                                               float&  t,
                                               float&  beta,
                                               float&  gamma )
{
  float3 e0 = p1 - p0;
  float3 e1 = p0 - p2;
  n  = cross( e0, e1 );

  float v   = dot( n, ray.direction );
  float r   = 1.0f / v;

  float3 e2 = p0 - ray.origin;
  float va  = dot( n, e2 );
  t         = r*va;

  if(t < ray.tmax && t > ray.tmin) {
    float3 i   = cross( e2, ray.direction );
    float v1   = dot( i, e1 );
    beta = r*v1;
    if(beta >= 0.0f){
      float v2 = dot( i, e0 );
      gamma = r*v2;
      return ( (v1+v2)*v <= v*v && gamma >= 0.0f );
    }
  }
  return false;
}


/*
  calculates refraction direction
  r   : refraction vector
  i   : incident vector
  n   : surface normal
  ior : index of refraction ( n2 / n1 )
  returns false in case of total internal reflection
*/
inline RT_HOSTDEVICE bool refract(float3& r, float3 i, float3 n, float ior)
{
  float3 nn = n;
  float negNdotV = dot(i,nn);
  float eta;

  if (negNdotV > 0.0f)
  {
    eta = ior;
    nn = -n;
    negNdotV = -negNdotV;
  }
  else
  {
    eta = 1.f / ior;
  }

  const float k = 1.f - eta*eta * (1.f - negNdotV * negNdotV);

  if (k < 0.0f) {
    return false;
  } else {
    r = normalize(eta*i - (eta*negNdotV + sqrtf(k)) * nn);
    return true;
  }
}

/* Schlick approximation of Fresnel reflectance */
inline RT_HOSTDEVICE float fresnel_schlick(float cos_theta, float exponent = 5.0f,
                                                            float minimum  = 0.0f,
                                                            float maximum  = 1.0f)
{
  /*
     clamp the result of the arithmetic due to floating point precision:
     the result should lie strictly within [minimum, maximum]
    return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta, exponent),
                 minimum, maximum);

  */

  /* The max doesn't seem like it should be necessary, but without it you get
     annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
  */
  return clamp(minimum + (maximum - minimum) * powf(fmaxf(0.0f,1.0f - cos_theta), exponent),
               minimum, maximum);
}

inline RT_HOSTDEVICE float3 fresnel_schlick(float cos_theta, float exponent,
                                            float3 minimum, float3 maximum)
{
  return make_float3(fresnel_schlick(cos_theta, exponent, minimum.x, maximum.x),
                     fresnel_schlick(cos_theta, exponent, minimum.y, maximum.y),
                     fresnel_schlick(cos_theta, exponent, minimum.z, maximum.z));
}


// Calculate the NTSC luminance value of an rgb triple
inline RT_HOSTDEVICE float luminance( const float3& rgb )
{
  const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
  return  dot( rgb, ntsc_luminance );
}

inline RT_HOSTDEVICE void cosine_sample_hemisphere( float u1, float u2, float3& p )
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}

// Maps concentric squares to concentric circles (Shirley and Chiu)
inline RT_HOSTDEVICE float2 square_to_disk( float2 sample )
{
  float phi, r;

  const float a = 2.0f * sample.x - 1.0f;
  const float b = 2.0f * sample.y - 1.0f;

  if (a > -b)
  {
    if (a > b)
    {
      r = a;
      phi = (float)M_PI_4f * (b/a);
    }
    else
    {
      r = b;
      phi = (float)M_PI_4f * (2.0f - (a/b));
    }
  }
  else
  {
    if (a < b)
    {
      r = -a;
      phi = (float)M_PI_4f * (4.0f + (b/a));
    }
    else
    {
      r = -b;
      phi = (b) ? (float)M_PI_4f * (6.0f - (a/b)) : 0.0f;
    }
  }

  return make_float2( r * cosf(phi), r * sinf(phi) );
}

// Convert cartesian coordinates to polar coordinates
inline RT_HOSTDEVICE float3 cart_to_pol(float3 v)
{
  float azimuth;
  float elevation;
  float radius = length(v);

  float r = sqrtf(v.x*v.x + v.y*v.y);
  if (r > 0.0f)
  {
    azimuth   = atanf(v.y / v.x);
    elevation = atanf(v.z / r);

    if (v.x < 0.0f)
      azimuth += M_PIf;
    else if (v.y < 0.0f)
      azimuth += M_PIf * 2.0f;
  }
  else
  {
    azimuth = 0.0f;

    if (v.z > 0.0f)
      elevation = +M_PI_2f;
    else
      elevation = -M_PI_2f;
  }

  return make_float3(azimuth, elevation, radius);
}

// Orthonormal basis
struct Onb
{
  inline RT_HOSTDEVICE Onb( const float3& normal )
    {
      m_normal = normal;

      if( fabs(m_normal.x) > fabs(m_normal.z) )
      {
        m_binormal.x = -m_normal.y;
        m_binormal.y =  m_normal.x;
        m_binormal.z =  0;
      }
      else
      {
        m_binormal.x =  0;
        m_binormal.y = -m_normal.z;
        m_binormal.z =  m_normal.y;
      }

      m_binormal = normalize(m_binormal);
      m_tangent = cross( m_binormal, m_normal );
    }

  inline RT_HOSTDEVICE void inverse_transform( float3& p )
    {
      p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
    }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

} // end namespace optix

/*
 * When looking for operators for a type, only the scope the type is defined in (plus the
 * global scope) is searched.  In order to make the operators behave properly we are
 * pulling them into the global namespace.
 */
#if defined(RT_PULL_IN_VECTOR_TYPES)
using optix::operator-;
using optix::operator-=;
using optix::operator+;
using optix::operator+=;
using optix::operator*;
using optix::operator*=;
using optix::operator/;
using optix::operator/=;
using optix::operator==;
#endif // #if defined(RT_PULL_IN_VECTOR_TYPES)

/*
 * Here are a list of functions that are overloaded in both the global and optix
 * namespace.  If they have a global namespace version, then the overloads in the optix
 * namespace need to be pulled in, so that all the overloads are on the same level.
 */

/* These are defined by CUDA in the global namespace */
#if defined(RT_PULL_IN_VECTOR_FUNCTIONS)
#define RT_DEFINE_HELPER(type) \
  using optix::make_##type##1; \
  using optix::make_##type##2; \
  using optix::make_##type##3; \
  using optix::make_##type##4;

/* Some types didn't exist until CUDA 3.0.  CUDA_VERSION isn't defined while
* building CUDA code, so we also need to check the CUDART_VERSION value. */
#if (CUDA_VERSION >= 3000) || (CUDART_VERSION >= 3000)
#define RT_DEFINE_HELPER2(type) RT_DEFINE_HELPER(type)
#else
#define RT_DEFINE_HELPER2(type) \
  using ::make_##type##1; \
  using ::make_##type##2; \
  using ::make_##type##3; \
  using ::make_##type##4; 
#endif


RT_DEFINE_HELPER(char)
RT_DEFINE_HELPER(uchar)
RT_DEFINE_HELPER(short)
RT_DEFINE_HELPER(ushort)
RT_DEFINE_HELPER(int)
RT_DEFINE_HELPER(uint)
RT_DEFINE_HELPER(long)
RT_DEFINE_HELPER(ulong)
RT_DEFINE_HELPER(float)
RT_DEFINE_HELPER2(longlong)
RT_DEFINE_HELPER2(ulonglong)
RT_DEFINE_HELPER2(double)

using ::make_double3x3;
using ::make_double4x4;

#undef RT_DEFINE_HELPER
#undef RT_DEFINE_HELPER2

#endif // #if defined(RT_PULL_IN_VECTOR_FUNCTIONS)

/* These are defined by CUDA and non-Windows platforms in the global namespace. */
#if !defined(_WIN32) || defined (__CUDACC__)
using optix::fmaxf;
using optix::fminf;
using optix::copysignf;
using optix::copy_sign;
#endif

/* These are always in the global namespace. */
using optix::expf;
using optix::floor;

/* These are defined by CUDA in the global namespace */
#if defined (__CUDACC__)
using optix::max;
using optix::min;
#endif



#endif // #ifndef __optixu_optixu_math_namespace_h__
