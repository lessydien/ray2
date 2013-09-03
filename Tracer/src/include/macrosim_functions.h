/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__MACROSIM_FUNCTIONS_H__)
#define __MACROSIM_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "host_defines.h"
#include "macrosim_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/





// static __inline__ __host__ __device__ double1 make_double1(double x)
// {
  // double1 t; t.x = x; return t;
// }

// static __inline__ __host__ __device__ double2 make_double2(double x, double y)
// {
  // double2 t; t.x = x; t.y = y; return t;
// }

// static __inline__ __host__ __device__ double3 make_double3(double x, double y, double z)
// {
  // double3 t; t.x = x; t.y = y; t.z = z; return t;
// }

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

// static __inline__ __host__ __device__ double4 make_double4(double x, double y, double z, double w)
// {
  // double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
// }


#endif /* !__MACROSIM_FUNCTIONS_H__ */
