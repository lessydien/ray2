
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

//typedef struct 
//{
//  double3 position;
//  double3 direction;
//  double t_hit;
//  float  flux;
//  int    depth;
//} rayStruct;

//struct PerRayData_shadow
//{
//  float3 attenuation;
//};


rtDeclareVariable(int,               max_depth, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
//rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
//rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );
rtDeclareVariable(gaussBeamRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,          geometryID, , );
rtDeclareVariable(gaussBeam_geometricNormal, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(gaussBeam_t, t_hit, attribute t_hit, );

__device__ void absorbingGaussBeam_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
  //if (prd.currentGeometryID==geometryID)
  //{
  //  rtIgnoreIntersection();
  //}
}

__device__ void absorbingGaussBeam_closestHit_device( gaussBeam_geometricNormal p_normal, gaussBeam_t t_hit )
{
  //double3 hit_point = prd.position + t_hit * prd.direction;
  //
  //// pass the position back up the tree
  //prd.position = hit_point;
  //prd.currentGeometryID=geometryID;
}
