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
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "../rayData.h"
#include "AsphericalSurface_GeomRender_Intersect.h"

rtDeclareVariable(AsphericalSurface_GeomRender_ReducedParams, params, , ); 
//rtDeclareVariable(int, materialListLength, , ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(geomRenderRayStruct, prd, rtPayload, ); // get per-ray-data structure
// variables that are communicated to the hit program via the attribute-mechanism
rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );


__forceinline__ __device__ Mat_hitParams calcHitParams(double t)
{
  return calcHitParamsAsphere(prd.position+t*prd.direction, params);
}

RT_PROGRAM void intersect(int)
{
		double t =  intersectRayAsphere(prd.position, prd.direction, params);
    	// check whether intersection is within valid interval of t
		if( rtPotentialIntersection( (float)t ))
		{
			// calc normal in intersection
			hitParams=calcHitParams(t);
			// save hit paramter
			t_hit=t;
			// pass geometryID to hit-program
			geometryID=params.geometryID;
			// call any hit function of the respective material
			rtReportIntersection( 0 );
			
		}

}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 maxBox=make_float3(0,0,0);
  float3 minBox=make_float3(0,0,0);
  aabb->set(minBox, maxBox);
}
