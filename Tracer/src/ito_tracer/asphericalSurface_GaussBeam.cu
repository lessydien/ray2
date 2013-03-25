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
#include "AsphericalSurface_Intersect.h"

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(AsphericalSurface_ReducedParams, params, , ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, ); // get per-ray-data structure
// variables that are communicated to the hit program via the attribute-mechanism
rtDeclareVariable(gaussBeam_geometricNormal, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(gaussBeam_t, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );


__forceinline__ __device__ gaussBeam_geometricNormal calcHitParams(gaussBeam_t t)
{
  gaussBeam_geometricNormal normal;
  normal.normal_baseRay=make_double3(0,0,0);
  normal.normal_waistRayX=make_double3(0,0,0);
  normal.normal_waistRayY=make_double3(0,0,0);
  normal.normal_divRayX=make_double3(0,0,0);
  normal.normal_divRayY=make_double3(0,0,0);
  return normal;

  // pass geometryID to hit-program
  geometryID=params.geometryID;
 
	//double3 n, hitPoint;
	//double h;
	//double	c=	params.c;	//paramteres of the asphere.
	//double	k=	params.k;
	//double	c2=	params.c2,
	//		c4=	params.c4,
	//		c6=	params.c6,
	//		c8=	params.c8,
	//		c10=params.c10;
	//double x,y;
	//
	//hitPoint=prd.position+t*prd.direction;
	//x=hitPoint.x;
	//y=hitPoint.y;
	//h=sqrt(x*x+y*y);
	//
	//if (h==0.0)
	//{
	//	n=params.orientation;
	//}
	//else
	//{
	//	n.x= ((c*h/(sqrt(1-(1+k)*c*c*h*h))+2*c2*pow(h,1)+4*c4*pow(h,3)+6*c6*pow(h,5)+8*c8*pow(h,7)+10*c10*pow(h,9))*(x/h));
	//	//n.y= ((c*h/(sqrt(1-(1+k)*c*c*h*h))+2*c2*pow(h,1)+4*c4*pow(h,3)+6*c6*pow(h,5)+8*c8*pow(h,7)+10*c10*pow(h,9))*(y/h));
	//	//faster: // n.y=n.x / (x/h) * (y/h)
	//	//n.y= n.x /x*y
	//	if (x==0.0)
	//	{
	//		n.y= ((c*h/(sqrt(1-(1+k)*c*c*h*h))+2*c2*pow(h,1)+4*c4*pow(h,3)+6*c6*pow(h,5)+8*c8*pow(h,7)+10*c10*pow(h,9))*(y/h));
	//	}
	//	else
	//	{
	//		n.y=n.x/x*y;
	//	}

	//n.z=-1.0;
	//n=normalize(n);
	//}
	//return n;
}

RT_PROGRAM void intersect(int)
{
	//double t =  intersectRayAsphere(prd.position, prd.direction, params);
	//if ( t!=0 )
	//{
	//  // check whether intersection is within valid interval of t
	//  if( rtPotentialIntersection( (float)t )&& checkAperture(params.vertex,params.orientation,prd.position+t*prd.direction,params.apertureType,params.apertureRadius))
	//  {
	//    // calc normal in intersection
	//    geometric_normal=calcHitParams(t);
	//    // save hit paramter
	//    t_hit=t;
	//    // call any hit function
	//	

	//    rtReportIntersection( 0 );
	//  }
	//}

}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(boxmin, boxmax);
}
