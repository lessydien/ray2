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
#include "../rayData.h"
#include "../rayTracingMath.h"
#include "Scatter_Lambert2D_GeomRender_hit.h"
#include "MaterialAbsorbing_GeomRender_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_GeomRender_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(ScatLambert2D_params, scatterParams, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(geomRenderRayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void absorbingScatLambert2DGeomRender_anyHit_device()
{
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }

}

__forceinline__ __device__ void absorbingScatLambert2DGeomRender_closestHit_device( Mat_GeomRender_hitParams hitParams, double t_hit )
{

  prd.position = prd.position + (t_hit) * prd.direction;
  prd.currentGeometryID=geometryID;

  rtPrintf("closest hit ID %i \n", geometryID);
  rtPrintf("flux %.20lf \n", prd.flux);
  hitAbsorbing(prd, hitParams, t_hit, geometryID); // has no effect anyway
  hitLambert2D_GeomRender(prd, hitParams, scatterParams);  
  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false;  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  absorbingScatLambert2DGeomRender_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  absorbingScatLambert2DGeomRender_closestHit_device( hitParams, t_hit ); 
}
