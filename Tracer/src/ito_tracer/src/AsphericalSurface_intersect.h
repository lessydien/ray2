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

/**\file AsphericalSurface_intersect.r
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef ASPHERICALSURFACEINTERSECT_H
  #define ASPHERICALSURFACEINTERSECT_H
  
/* include header of basis class */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"

/* declare class */
/**
  *\class   AsphericalSurface_Params 
  *\ingroup Geometry
  *\brief   reduced set of params that is calculated before the actual tracing from the full set of params. This parameter set will be loaded onto the GPU if the tracing is done there
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class AsphericalSurface_ReducedParams : public Geometry_ReducedParams
{
  public:
  double3 vertex;
  double3 orientation;
  double k; // conic constant
  double c; // 1/(radius of curvature)
  double c2;
  double c4;
  double c6;
  double c8;
  double c10;
  double c12;
  double c14;
  double c16;
//  double rotNormal; // rotation of geometry around its normal
  //int    geometryID;

  ApertureType apertureType;
  double2 apertureRadius;
};

/**
 * \detail calcHitParamsAsphere 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 * The idea is to calculate the intersection of the surface normal with the optical axis of the surface from a 
 * formula from Malacara, Handbook of OpticalDesign, 2nd ed, A.2.1.1. Then the normal at the given point is simply 
 * the normalized vector connecting the given point and this point of intersection...
 *
 *
 * \param[in] double3 position,AsphericalSurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsAsphere(double3 position,AsphericalSurface_ReducedParams params)
{

	double3 n;
	double r;

	double3 offsetVec, offsetVecParallel, offsetVecOrthogonal;//new
	
	//r=sqrt((x-params.vertex.x)*(x-params.vertex.x)+(y-params.vertex.y)*(y-params.vertex.y));//old
	
	//for calculating the tangetial plane of the asphere at the new point x1,y1,z1
	//we need. dz/dx and dz/dy.
	//we use: dz/dx=(dz/dh)*(dh/dx); likewise for dz/dy
	//dh/dx=1/(2*sqrt(r) * 2x = x/r
	//normal vektor at x1,y1,z1 = 1/lenght*(dz/dx,dz/dy,-1)
	//dz/dh= params.c*r/(sqrt(1-(1+k)*c^2*r^2)+2*c2*r^1+4*c4+r^3+6*c6*r^5+8*c8*r^7+10*c10*r^9

	offsetVec=position-params.vertex;//new
	offsetVecParallel=dot(offsetVec,params.orientation)*params.orientation;//new
	offsetVecOrthogonal=offsetVec-offsetVecParallel;//new
	r=length(offsetVecOrthogonal);//new

	// calc derivative of z at given radial distance
	// see H. Gross, Handbook of optical systems, Vol 1, pp. 198
	double dzdr=params.c*r/(sqrt(1-(1+params.k)*params.c*params.c*r*r))+2*params.c2*pow(r,1)+4*params.c4*pow(r,3)+6*params.c6*pow(r,5)+8*params.c8*pow(r,7)+10*params.c10*pow(r,9)+12*params.c12*pow(r,11)+14*params.c14*pow(r,13)+16*params.c16*pow(r,15);

	// calc aberration of normal
	// see Malacara, Handbook of OpticalDesign, 2nd ed, A.2.1.1
	double Ln=r/dzdr+(params.c*r*r/(1+sqrt(1-(1+params.k)*params.c*params.c*r*r))+ params.c2*pow(r,2)+params.c4*pow(r,4)+params.c6*pow(r,6)+params.c8*pow(r,8)+params.c10*pow(r,10)+params.c12*pow(r,12)+params.c14*pow(r,14)+params.c16*pow(r,16));
	double3 debugFocus=params.vertex+Ln*params.orientation;
	double3 debugPos=position;
	double3 debugOrientation=params.orientation;
	Mat_hitParams t_hitParams;
	t_hitParams.normal=normalize(params.vertex+Ln*params.orientation-position);

	//if (r==0.0)
	//{
	//	n=params.orientation;
	//}
	//else
	//{
	//	n.x= ((params.c*r/(sqrt(1-(1+params.k)*params.c*params.c*r*r))+2*params.c2*pow(r,1)+4*params.c4*pow(r,3)+6*params.c6*pow(r,5)+8*params.c8*pow(r,7)+10*params.c10*pow(r,9))*(position.x/r));
	//	//n.y= ((c*r/(sqrt(1-(1+k)*c*c*r*r))+2*c2*pow(r,1)+4*c4*pow(r,3)+6*c6*pow(r,5)+8*c8*pow(r,7)+10*c10*pow(r,9))*(y/r));
	//	//faster: // n.y=n.x / (x/r) * (y/r)
	//	//n.y= n.x /x*y
	//	if (position.x==0.0)
	//	{
	//		n.y= ((params.c*r/(sqrt(1-(1+params.k)*params.c*params.c*r*r))+2*params.c2*pow(r,1)+4*params.c4*pow(r,3)+6*params.c6*pow(r,5)+8*params.c8*pow(r,7)+10*params.c10*pow(r,9))*(position.y/r));
	//	}
	//	else
	//	{
	//		n.y=n.x/position.x*position.y;
	//	}

	//n.z=-1.0;
	//n=normalize(n);
	//}
	//Mat_hitParams t_hitParams;
	//t_hitParams.normal=-n;

	return t_hitParams;

}

/**
 * \detail intersectRayAsphere 
 *
 * calculates the intersection of a ray with an aspherical surface. The idea is to intersect the incoming ray with the plane that contains the surface vertex and that is normal to its orientation vector
 * Then the intersection point is projected along the orientation vector of the surface onto the real aspheric surface. If the distance of this projection is too large, the intersection of the incoming ray
 * and the plane beeing tangenial to the aspheric surface at the projected intersection is calculated. This is done iteratively until the distance of the projection is smaller than a predefined tolerance
 *
 * \param[in] double3 rayPosition, double3 rayDirection, AsphericalSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayAsphere(double3 rayPosition, double3 rayDirection, AsphericalSurface_ReducedParams params)
{
	double3 p0;		//intersection point of ray and vertex plane
	double3 p1;		//projected point
	double3 p2;		//intersection of ray with tangential plane at p1

	const double tolerance= 1e-10;//*pow(10.0,-10);	//maximum distanz between two successive calculated intersection points
	double	distance;	//distance between two succesive calculated points p1,p2
	double	t;			//return parameter
	double tHelp1, tHelp2;
	double	r;			//radial distance to asphere vertex (i.e. r in z=c1*r+c2*r^2+...)
	//possible it might be faster to calculate only h_squared=(x^2+y^2) and use that for further calculation

	double3	n;			//normal vektor of asphere plane at point p1

	double t0;				// parameter in ray equation. x=StartPoint+ t0* Direction

	Mat_hitParams t_hitParams; // hitParams of asphere at point p1
	AsphericalSurface_ReducedParams debugParams=params;

	////*****************************************
	double3 rayPos,rayDir,vertex,orientation;
	rayPos=rayPosition;
	rayDir=rayDirection;
	vertex = params.vertex;
	orientation=params.orientation;
	
	double3 offsetVec, offsetVecParallel, offsetVecOrthogonal;//new

	////calculating intersection with vertex plane
	t0=intersectRayPlane(rayPosition, rayDirection, params.vertex, params.orientation);
//	tHelp1 = dot((rayPosition-params.vertex), params.orientation);
//	tHelp2 = dot(rayDirection, params.orientation); 
//	t0= -tHelp1/tHelp2;
	p0=rayPosition+t0*rayDirection;

	//projecting the intersection of the ray with the vertex plane to the asphere
	//r=sqrt((p0.x-params.vertex.x)*(p0.x-params.vertex.x)+(p0.y-params.vertex.y)*(p0.y-params.vertex.y)); //old
	//p1.x= p0.x;//old
	//p1.y= p0.y;//old
	
	offsetVec=p0-params.vertex;//new
	//calculates the distance to the orientation vector
	offsetVecParallel=dot(offsetVec,params.orientation)*params.orientation;//new
	offsetVecOrthogonal=offsetVec-offsetVecParallel;//new
	r=length(offsetVecOrthogonal);//new

	//if asphere is undefined
	if ((1-(1+params.k)*params.c*params.c*r*r)<0.0)
	{
		return 0;
	}
	//<5.88>
	//calculating coresponding z-value to the x and y value
	//p1.z=p0.z+ c*r*r/(1+sqrt(1-(1+k)*c*c*r*r))+ c2*pow(r,2)+c4*pow(r,4)+c6*pow(r,6)+c8*pow(r,8)+c10*pow(r,10);
	//p1=p0.z+ params.orientation*(params.c*r*r/(1+sqrt(1-(1+params.k)*params.c*params.c*r*r))+ params.c2*pow(r,2)+params.c4*pow(r,4)+params.c6*pow(r,6)+params.c8*pow(r,8)+params.c10*pow(r,10)+params.c12*pow(r,12)+params.c14*pow(r,14));//new
	p1=p0 + params.orientation*(params.c*r*r/(1+sqrt(1-(1+params.k)*params.c*params.c*r*r))+ params.c2*pow(r,2)+params.c4*pow(r,4)+params.c6*pow(r,6)+params.c8*pow(r,8)+params.c10*pow(r,10)+params.c12*pow(r,12)+params.c14*pow(r,14)+params.c16*pow(r,16));//new

	int counter=0;
	do{
		//<5.61,62,90,91>
		//r=sqrt((p1.x-params.vertex.x)*(p1.x-params.vertex.x)+(p1.y-params.vertex.y)*(p1.y-params.vertex.y));//old
		
//		offsetVec=p1-params.vertex;//new//###
//		offsetVecParallel=dot(offsetVec,params.orientation)*params.orientation;//new
//		offsetVecOrthogonal=offsetVec-offsetVecParallel;//new
//		r=length(offsetVecOrthogonal);//new

		t_hitParams=calcHitParamsAsphere(p1,params);
		//n=t_hitParams.normal;
		
		//calculating intersection of tangential plane and ray, then projecting to asphere

		t0=intersectRayPlane(rayPosition, rayDirection, p1, t_hitParams.normal);
		//tHelp1 = dot((rayPosition-p1), n);
		//tHelp2 = dot(rayDirection, n); 
		//t0= -tHelp1/tHelp2;
		p2=rayPosition+t0*rayDirection;

		// calculate projection to asphere
		//r=sqrt((p2.x-params.vertex.x)*(p2.x-params.vertex.x)+(p2.y-params.vertex.y)*(p2.y-params.vertex.y));//old
		//offsetVec=p0-params.vertex;//new
		//offsetVecParallel=dot(offsetVec,params.orientation)*params.orientation;//new
		//offsetVecOrthogonal=offsetVec-offsetVecParallel;//new
		//r=length(offsetVecOrthogonal);//new
		offsetVec=p2-params.vertex;//new
		offsetVecParallel=dot(offsetVec,params.orientation)*params.orientation;//new
		offsetVecOrthogonal=offsetVec-offsetVecParallel;//new
		r=length(offsetVecOrthogonal);//new


		//p2.z= p0.z+(c*r*r/(1+sqrt(1-(1+k)*c*c*r*r)))+ c2*pow(r,2)+c4*pow(r,4)+c6*pow(r,6)+c8*pow(r,8)+c10*pow(r,10);
		//p2=p0+ params.orientation*(params.c*r*r/(1+sqrt(1-(1+params.k)*params.c*params.c*r*r))+ params.c2*pow(r,2)+params.c4*pow(r,4)+params.c6*pow(r,6)+params.c8*pow(r,8)+params.c10*pow(r,10)+params.c12*pow(r,12)+params.c14*pow(r,14));//new
		double3 p3=p2-offsetVecParallel+params.orientation*(params.c*r*r/(1+sqrt(1-(1+params.k)*params.c*params.c*r*r))+ params.c2*pow(r,2)+params.c4*pow(r,4)+params.c6*pow(r,6)+params.c8*pow(r,8)+params.c10*pow(r,10)+params.c12*pow(r,12)+params.c14*pow(r,14)+params.c16*pow(r,16));//new

		p1= p3;
		
		//if (length(p2-p1)<tolerance) break; //accuracy reached
		if (length(p3-p2)<tolerance) break; //accuracy reached
		if (counter>50) return 0;
		counter++;

	}while(1);

	//calculating return value.
	//	pi= Intersection Point
	//	pi=rayPosition+t*rayDirection
	//	Function returns the paramter t

	// negative values of t are invalid
	if ( dot(p1-rayPosition,rayDirection) < 0 )
		return 0;

	t= length(p1-rayPosition);

	if (!checkAperture(params.vertex,params.tilt,rayPosition+t*rayDirection,params.apertureType,params.apertureRadius))
	{
		return 0;
	}
	return t;
}


#endif
