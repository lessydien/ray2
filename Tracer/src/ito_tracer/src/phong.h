
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
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


#include <optix.h>
#include <optix_math.h>
#include <sutilCommonDefines.h>
#include "helpers.h"

struct PerRayData_position
{
  float3 position;
  float  flux;
  int    depth;
};

//struct PerRayData_shadow
//{
//  float3 attenuation;
//};


rtDeclareVariable(int,               max_depth, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
//rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_position, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );

__device__ void phongShadowed()
{
  // this material is opaque, so it fully attenuates all shadow rays
  //prd_shadow.attenuation = make_float3(0);
  rtTerminateRay();
}

__device__ void phongShade( float3 p_normal,
                            float p_reflectivity )
{
  float3 hit_point = ray.origin + t_hit * ray.direction;
  //float3 hit_point;
  //hit_point.x=300.0f;
  //hit_point.y=2.0f;
  //hit_point.z=2.33f;
  
  // ambient contribution

  float flux;
  int depth;

  // ray tree attenuation
  flux = prd.flux * p_reflectivity;
  depth = prd.depth + 1;

  //// reflection ray
  //if( new_prd.flux >= 0.01f && new_prd.depth <= max_depth) {
  //  float3 R = reflect( ray.direction, p_normal );
  //  optix::Ray refl_ray = optix::make_Ray( hit_point, R, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
  //  rtTrace(top_object, refl_ray, new_prd);
  //  result += p_reflectivity * new_prd.result;
  //}
  
  // pass the color back up the tree
  prd.position = hit_point;
  prd.flux = flux;
  prd.depth = depth;
}
