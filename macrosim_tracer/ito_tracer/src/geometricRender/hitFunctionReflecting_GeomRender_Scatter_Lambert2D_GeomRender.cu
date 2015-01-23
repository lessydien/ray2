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
#include "MaterialReflecting_GeomRender_hit.h"
#include "Scatter_Lambert2D_GeomRender_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_GeomRender_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(geomRenderRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
//rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(ScatLambert2D_params, scatterParams, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void reflectingScatterLambert2DGeomRender_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void reflectingScatterLambert2DGeomRender_closestHit_device( Mat_GeomRender_hitParams hitParams, double t_hit )
{
	  
  hitReflecting(prd, hitParams, t_hit, geometryID, true);

  // create secondary ray
  geomRenderRayStruct sdRay=prd;
  sdRay.cumFlux=0;
  sdRay.secondary=true;
  sdRay.secondary_nr++;

  if (hitLambert2D_GeomRender(sdRay, hitParams, scatterParams)) // do the scattering
  {
      if (sdRay.secondary_nr<2 && sdRay.flux>1e-8)
      {
          double scene_epsilon=0.00001;
          optix::Ray ray = optix::make_Ray(make_float3(sdRay.position), make_float3(sdRay.direction), 0, scene_epsilon, RT_DEFAULT_MAX);
          for(;;)
          {
              rtTrace(top_object, ray, sdRay);
              // update ray
	          ray.origin=make_float3(sdRay.position);
	          ray.direction=make_float3(sdRay.direction);

              if (!sdRay.running)
                  break;
          }
          prd.cumFlux+=sdRay.cumFlux;
          prd.currentSeed=sdRay.currentSeed;
      }

  }
  prd.running=false;
  // continue primary ray
  //ScatLambert2D_params primParams=scatterParams;
  //primParams.impAreaType=AT_INFTY; // primary ray does not use the importance area

  //if (!hitLambert2D_GeomRender(prd, hitParams, primParams))
  //    prd.running=false;

  //if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	 // prd.running=false;
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  reflectingScatterLambert2DGeomRender_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  reflectingScatterLambert2DGeomRender_closestHit_device( hitParams, t_hit );
}
