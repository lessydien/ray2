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
#include "microLensArraySurface_Intersect.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, ); // get per-ray-data structure
rtDeclareVariable(MicroLensArraySurface_ReducedParams, params, , ); // centre of spherical surface
//rtDeclareVariable(int, materialListLength, , ); 
// variables that are communicated to hit program vie attribute mechanism
rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, );
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );

__forceinline__ __device__ Mat_hitParams calcHitParams(double t)
{
  return calcHitParamsMicroLensArraySurface(prd.position+t*prd.direction, params);
}

RT_PROGRAM void intersect(int)
{ 

	double t=intersectRayMicroLensArraySurface(prd.position,prd.direction,params);
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
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  double3 l_ex=make_double3(1,0,0);
  rotateRay(&l_ex,params.tilt);
  double3 l_ey=make_double3(0,1,0);
  rotateRay(&l_ey,params.tilt);
  // sag of lens
  double lensSag=copy_sign( (double)1.0, params.microLensRadius )*abs(params.microLensRadius-sqrt(params.microLensRadius*params.microLensRadius-params.microLensAptRad*params.microLensAptRad));
  float3 maxBox=make_float3(params.root+params.normal*lensSag+params.apertureRadius.x*l_ex+params.apertureRadius.y*l_ey);
  float3 minBox=make_float3(params.root-params.apertureRadius.x*l_ex-params.apertureRadius.y*l_ey);
  aabb->set(minBox, maxBox);
}
