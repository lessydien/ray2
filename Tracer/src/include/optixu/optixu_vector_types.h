
/*
 * Copyright (c) 2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#ifndef __optixu_optixu_vector_types_h__
#define __optixu_optixu_vector_types_h__

#include <limits.h>
#include <stddef.h>

#if defined(__cplusplus)

/*
 * We need to check to see if <vector_types.h> has already been included before we've
 * had a chance to include it here.  If, so that means that the contents will be in the
 * global namespace, and we will add them to the optix namespace below in order to have
 * overloaded function in the optix namespace function correctly
 */

#  if defined(__VECTOR_TYPES_H__)
#    define RT_PULL_IN_VECTOR_TYPES
#  endif

#include "macrosim_types.h"
#include "macrosim_functions.h"

namespace optix {

#endif /* #if defined (__cplusplus) */

#include "vector_types.h"

#if defined(__cplusplus)
} /* end namespace optix */
#endif

/* Pull the global namespace CUDA functions into the optix namespace. */
#if defined(RT_PULL_IN_VECTOR_TYPES)
#define RT_DEFINE_HELPER(type) \
  using ::type##1; \
  using ::type##2; \
  using ::type##3; \
  using ::type##4;

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

namespace optix {
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

using ::dim3;
using ::double3x3;
using ::make_double3x3;
using ::double4x4;
using ::make_double4x4;

} /* end namespace optix */

#undef RT_DEFINE_HELPER
#undef RT_DEFINE_HELPER2

#endif

#endif /* #ifndef __optixu_optixu_vector_types_h__ */

