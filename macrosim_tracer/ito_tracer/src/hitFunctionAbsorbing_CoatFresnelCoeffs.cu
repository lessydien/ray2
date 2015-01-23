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

#include <optix.h>
#include <optix_math.h>
//#include <commonStructs.h>
//#include "helpers.h"
#include "rayData.h"
#include "MaterialAbsorbing_hit.h"
#include "Coating_FresnelCoeffs_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
//rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_FresnelCoeffs_ReducedParams, coating_params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void absorbingCoatFresnelCoeffs_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
//  if (prd.currentGeometryID==geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void absorbingCoatFresnelCoeffs_closestHit_device( Mat_hitParams hitParams, double t_hit )
{
  rtPrintf("closest hit ID %i \n", geometryID);
  rtPrintf("flux %.20lf \n", prd.flux);
  hitAbsorbing(prd, hitParams, t_hit,geometryID);
  // if coating wants a reflection we keep the ray running and reflect it
  if (hitCoatingFresnelCoeff(prd, hitParams, coating_params))
	  prd.direction=reflect(prd.direction, hitParams.normal);
  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false;	
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  absorbingCoatFresnelCoeffs_anyHit_device();
}


RT_PROGRAM void closestHit()
{
//  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
//  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

//  float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  absorbingCoatFresnelCoeffs_closestHit_device( hitParams, t_hit );
}
