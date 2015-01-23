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
#include "../rayData.h"
#include "../rayTracingMath.h"
#include "Scatter_TorranceSparrow2D_GeomRender_hit.h"
#include "../MaterialReflecting_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(ScatTorranceSparrow2D_params, scatterParams, , ); 
rtDeclareVariable(MatReflecting_params, params, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(geomRenderRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void reflectingScatTorrSparr2D_anyHit_device()
{
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void reflectingScatTorrSparr2D_closestHit_device( Mat_hitParams hitParams, double t_hit )
{
	// reflect ray
	hitReflecting(prd, hitParams, t_hit, geometryID, true);

	//// we have a geomRender ray here ... hopefully ...
	//if (!ray_interpreted->secondary)
	//{
	//	// create secondary ray
	//	rayStruct_PathTracing l_ray;
	//	l_ray.position=ray_interpreted->position;
	//	l_ray.depth=ray_interpreted->depth;
	//	l_ray.currentGeometryID=ray_interpreted->currentGeometryID;
	//	l_ray.currentSeed=ray_interpreted->currentSeed;
	//	l_ray.flux=ray_interpreted->flux;
	//	l_ray.lambda=ray_interpreted->lambda;
	//	l_ray.nImmersed=ray_interpreted->nImmersed;
	//	l_ray.opl=ray_interpreted->opl;
	//	l_ray.result=ray_interpreted->result;
	//	l_ray.running=true;
	//	// aim it towards light source
	//	aimRayTowardsImpArea(l_ray.direction, l_ray.position, scatterParams.impAreaRoot, scatterParams.impAreaHalfWidth, scatterParams.impAreaTilt, scatterParams.impAreaType, l_ray.currentSeed);
	//	// adjust flux of secondary ray according to scattering angle
	//	double l_scatAngle=dot(l_ray.direction,ray_interpreted->direction);
	//	l_ray.flux=l_ray.flux*scatterParams.Kdl*cos(l_scatAngle)+scatterParams.Ksl*exp(-l_scatAngle*l_scatAngle/(2*scatterParams.sigmaXsl))+scatterParams.Ksp*exp(-l_scatAngle*l_scatAngle/(2*scatterParams.sigmaXsp));
	//	// trace secondary ray only if it is not blocked by the scattering surface itself
	//	if ( dot(ray_interpreted->direction,l_ray.direction)>0 )
	//	{
	//		optix::Ray l_OptiXray = optix::make_Ray(make_float3(0.0f), make_float3(0.0f), 0, 0.1f, RT_DEFAULT_MAX);
	//		// trace secondary ray towards light source
	//		for(;;) 
	//		{			
	//			rtTrace(top_object, l_OptiXray, l_ray);
	//			if (!l_ray.running)
	//				break;
	//		}
	//	}
	//	// add result of secondary ray to our initial ray
	//	ray_interpreted->result=ray_interpreted->result+l_ray.result; // do we need some kind of normalization here ???!!!!
	//}
	//// scatter ray
	//hitTorranceSparrow2D_GeomRender(*ray_interpreted, hitParams, scatterParams);

 // if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	//  prd.running=false;  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  reflectingScatTorrSparr2D_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  reflectingScatTorrSparr2D_closestHit_device( hitParams, t_hit ); 
}
