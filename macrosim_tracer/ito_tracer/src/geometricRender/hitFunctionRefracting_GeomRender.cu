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
#include "MaterialRefracting_GeomRender_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_GeomRender_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int, geometryID, attribute geometryID , );
rtDeclareVariable(MatRefracting_params, params, , ); 
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(geomRenderRayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void refractingGeomRender_anyHit_device()
{
  // this material is opaque, so it fully attenuates all shadow rays
  //prd_shadow.attenuation = make_float3(0);
  //rtTerminateRay();
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }

}

__forceinline__ __device__ void refractingGeomRender_closestHit_device( Mat_GeomRender_hitParams hitParams, double t_hit )
{
  if (prd.depth < max_depth)
  {
    rtPrintf("closest hit ID %i \n", geometryID);
	//rtPrintf("flux %.20lf \n", prd.flux);
    bool coat_reflected =false;
    hitRefracting(prd, hitParams, params, t_hit, geometryID, coat_reflected);
//	rtPrintf("position %lf %lf %lf\n", prd.position.x, prd.position.y, prd.position.z);
  }
  else
  {
    rtPrintf("ray stopped in hitRefracting!!");
    prd.running=false; // stop ray
  }
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  rtPrintf("any hit ID %i \n", geometryID);
  refractingGeomRender_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  refractingGeomRender_closestHit_device( hitParams, t_hit );
}
