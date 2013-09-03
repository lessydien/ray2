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

#if !defined(__MACROSIM_TYPES_H__)
#define __MACROSIM_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "vector_types.h"
#include <nppversion.h>

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if !defined(__cuda_assign_operators)

#define __cuda_assign_operators(tag)

#endif /* !__cuda_assign_operators */

#if !defined(__CUDACC__) && !defined(__CUDABE__) && \
    defined(_WIN32) && !defined(_WIN64)

#pragma warning(push)
#pragma warning(disable: 4201 4408)

#define __cuda_builtin_vector_align8(tag, members) \
struct __device_builtin__ tag                      \
{                                                  \
    union                                          \
    {                                              \
        struct { members };                        \
        struct { long long int :1,:0; };           \
    };                                             \
}

#else /* !__CUDACC__ && !__CUDABE__ && _WIN32 && !_WIN64 */

#define __cuda_builtin_vector_align8(tag, members) \
struct __device_builtin__ __align__(8) tag         \
{                                                  \
    members                                        \
}

#endif /* !__CUDACC__ && !__CUDABE__ && _WIN32 && !_WIN64 */

#if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < 4200)
    
     /*DEVICE_BUILTIN*/
     struct double1
     {
       double x;
       __cuda_assign_operators(double1)
     };

     /*DEVICE_BUILTIN*/
     typedef struct double1 double1;


     /*DEVICE_BUILTIN*/
     struct __builtin_align__(16) double2
     {
       double x, y;
       __cuda_assign_operators(double2);
 
     };

     /*DEVICE_BUILTIN*/
     typedef struct double2 double2;

     /*DEVICE_BUILTIN*/
     struct double3
     {
       double x, y, z;
       __cuda_assign_operators(double3)
     };

     /*DEVICE_BUILTIN*/
     typedef struct double3 double3;


     /*DEVICE_BUILTIN*/
     struct __builtin_align__(16) double4
     {
       double x, y, z, w;
       __cuda_assign_operators(double4)
     };
     /*DEVICE_BUILTIN*/
     typedef struct double4 double4;
 #endif

/*DEVICE_BUILTIN*/
struct double4x4
{
	double m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44;
	__cuda_assign_operators(double4x4)
};

/*DEVICE_BUILTIN*/
struct double3x3
{
	double m11, m12, m13, m21, m22, m23, m31, m32, m33;
	__cuda_assign_operators(double3x3)
};

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/





/*DEVICE_BUILTIN*/
typedef struct double4x4 double4x4;
/*DEVICE_BUILTIN*/
typedef struct double3x3 double3x3;

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/


#undef  __cuda_assign_operators
#undef  __cuda_builtin_vector_align8




/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* !__MACROSIM_TYPES_H__ */
