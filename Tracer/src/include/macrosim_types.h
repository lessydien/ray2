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

/* Some types didn't exist until CUDA 3.0.  CUDA_VERSION isn't defined while
* building CUDA code, so we also need to check the CUDART_VERSION value. */
//#if (CUDA_VERSION < 3000) //|| (CUDART_VERSION < 3000)
#if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < 4000)
    
     *DEVICE_BUILTIN*/
     struct double1
     {
       double x;
       __cuda_assign_operators(double1)
     };

     *DEVICE_BUILTIN*/
     typedef struct double1 double1;


     *DEVICE_BUILTIN*/
     struct __builtin_align__(16) double2
     {
       double x, y;
       __cuda_assign_operators(double2);
 
     };

     *DEVICE_BUILTIN*/
     typedef struct double2 double2;

     *DEVICE_BUILTIN*/
     struct double3
     {
       double x, y, z;
       __cuda_assign_operators(double3)
     };

     *DEVICE_BUILTIN*/
     typedef struct double3 double3;


     *DEVICE_BUILTIN*/
     struct __builtin_align__(16) double4
     {
       double x, y, z, w;
       __cuda_assign_operators(double4)
     };
     *DEVICE_BUILTIN*/
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

typedef struct double3x3 blubb;

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
