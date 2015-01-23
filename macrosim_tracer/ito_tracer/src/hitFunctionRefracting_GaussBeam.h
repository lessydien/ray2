
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
#include <optix_math_new.h>
#include <sutilCommonDefines.h>
#include "helpers.h"
#include "rayData.h"
#include "rayTracingMath.h"


rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(gaussBeamRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,          geometryID, , );
rtDeclareVariable(gaussBeam_geometricNormal, geometric_normal, attribute geometric_normal, ); // normal to the geometry at the hit-point. at a plane surface this will simply be the normal of the definition of the plane surface
rtDeclareVariable(gaussBeam_t, t_hit, attribute t_hit, ); 

__device__ void refractingGaussBeam_anyHit_device()
{
  // this material is opaque, so it fully attenuates all shadow rays
  //prd_shadow.attenuation = make_float3(0);
  //rtTerminateRay();
  //if (prd.currentGeometryID == geometryID)
  //{
  //  rtIgnoreIntersection();
  //}

}

__device__ void refractingGaussBeam_closestHit_device( gaussBeam_geometricNormal p_normal, gaussBeam_t t_hit )
{
 // double3 hit_point = prd.position + (t_hit) * prd.direction;
 // prd.currentGeometryID=geometryID;

 // if (prd.depth < max_depth)
 // {
 //   double3 dirRefr=calcSnellsLaw(prd.direction, p_normal, (double)1.0, (double)1.3);
 //   // refracted ray
	//optix::Ray refr_ray = optix::make_Ray( make_float3(hit_point), make_float3(dirRefr), 0, scene_epsilon, RT_DEFAULT_MAX );
 //   rayStruct new_prd;
	//new_prd=prd;
 //   new_prd.position=hit_point;
 //   new_prd.direction=dirRefr;
	//new_prd.depth=prd.depth+1;

 //   rtTrace(top_object, refr_ray, new_prd);
 //   // pass the data back up the tree
	//prd=new_prd;
 // }
 // else
 // {
 //   prd.position=hit_point;
 // }
 // 
}
