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
#include "rayData.h"
#include "ParabolicSurface_Intersect.h"

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
//rtDeclareVariable(simMode, mode, , );
rtDeclareVariable(rayStruct, prd, rtPayload, ); // get per-ray-data structure
rtDeclareVariable(ParabolicSurface_ReducedParams, params, , ); // centre of spherical surface
//rtDeclareVariable(int, materialListLength, , ); 
// variables that are communicated to hit program vie attribute mechanism
rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, );
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );

__forceinline__ __device__ Mat_hitParams calcHitParams(double t)
{
  return calcHitParamsParabol(prd.position+t*prd.direction, params);
}

RT_PROGRAM void intersect(int)
{ 
	double t;
	// we only calculate the intersection in nonsequential mode or if the current geometry is the next to intersect in "sequential mode"
//	if ( (mode==SIM_GEOMRAYS_NONSEQ) || (prd.currentGeometryID==params.geometryID-1) )
//	{
		// if the radius is zero in both direction we use the plane surface intersection instead
		if ( (params.curvatureRadius.x==0) && (params.curvatureRadius.y==0) )
		{
			double3 normal=params.orientation;
			double3 root=params.centre;
			t = intersectRayPlane(prd.position, prd.direction, root, normal);
		}
		else
		{
			t = intersectRayParabol(prd.position, prd.direction, params);
		}
		// check wether intrersection is within valid interval of t
		if( rtPotentialIntersection( (float)t ) ) 
		{
			//rtPrintf("normal %.20lf %.20lf %.20lf \n", params.orientation.x, params.orientation.y, params.orientation.z);
			
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
