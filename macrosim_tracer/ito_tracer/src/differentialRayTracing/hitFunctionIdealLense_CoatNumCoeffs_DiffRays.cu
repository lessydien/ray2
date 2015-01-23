
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
//#include <commonStructs.h>
//#include "helpers.h"
#include "../rayData.h"
#include "MaterialIdealLense_DiffRays_hit.h"
#include "Coating_NumCoeffs_DiffRays_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_DiffRays_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(diffRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_NumCoeffs_ReducedParams, coating_params, , );
rtDeclareVariable(MatIdealLense_DiffRays_params,               params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__device__ void absorbingCoatNumCoeffs_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
  if (prd.currentGeometryID==geometryID)
  {
    rtIgnoreIntersection();
  }
}

__device__ void absorbingCoatNumCoeffs_closestHit_device( Mat_DiffRays_hitParams hitParams, double t_hit )
{
  rtPrintf("closest hit ID %i \n", geometryID);
  rtPrintf("flux %.20lf \n", prd.flux);
  bool coat_reflected=hitCoatingNumCoeff(prd, hitParams, coating_params);
  hitIdealLense_DiffRays(prd, hitParams, params, t_hit, geometryID, coat_reflected);
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  absorbingCoatNumCoeffs_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  absorbingCoatNumCoeffs_closestHit_device( hitParams, t_hit );
}
