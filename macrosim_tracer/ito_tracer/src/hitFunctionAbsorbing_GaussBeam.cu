
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

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(gaussBeamRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,          geometryID, , );
rtDeclareVariable(gaussBeam_geometricNormal, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(gaussBeam_t, t_hit, attribute t_hit, );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

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
	// check wether all the other rays hit the surface as well 
	if ( (t_hit.t_waistRayX!=0)&&(t_hit.t_waistRayY!=0)&&(t_hit.t_divRayX!=0)&&(t_hit.t_divRayY != 0) ) 
	{
		
	}
	else
	{
		// terminate the ray with an error
	}
  //double3 hit_point = prd.position + t_hit * prd.direction;
  //
  //// pass the position back up the tree
  //prd.position = hit_point;
  //prd.currentGeometryID=geometryID;
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  absorbingGaussBeam_anyHit_device();
}


RT_PROGRAM void closestHit()
{
//  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
//  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

//  float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  absorbingGaussBeam_closestHit_device( geometric_normal, t_hit );
}
