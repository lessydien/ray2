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
#include "AsphericalSurface_DiffRays_Intersect.h"

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(AsphericalSurface_DiffRays_ReducedParams, params, , ); 
//rtDeclareVariable(int, materialListLength, , ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(diffRayStruct, prd, rtPayload, ); // get per-ray-data structure
//rtDeclareVariable(simMode, mode, , );
// variables that are communicated to the hit program via the attribute-mechanism
rtDeclareVariable(Mat_DiffRays_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );


__forceinline__ __device__ Mat_DiffRays_hitParams calcHitParams(double t)
{
  return calcHitParamsAsphere_DiffRays(prd.position+t*prd.direction, params);
}

RT_PROGRAM void intersect(int)
{
	// we only calculate the intersection in nonsequential mode or if the current geometry is the next to intersect in "sequential mode"
	//(ray.currentGeometryID==this->geometryID)
//	if ( ((mode==SIM_GEOMRAYS_SEQ)&&(mode==SIM_DIFFRAYS_SEQ)&&(prd.currentGeometryID == params.geometryID-1)) || ((mode==SIM_GEOMRAYS_NONSEQ)&&(prd.currentGeometryID != params.geometryID)) )
//	{
		double t =  intersectRayAsphere_DiffRays(prd.position, prd.direction, params);
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
//	}

}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(boxmin, boxmax);
}
