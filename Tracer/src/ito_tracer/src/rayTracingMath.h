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

/**\file rayTracingMath.h
* \brief collection of functions that are commonly used in raytracing. These functions are used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef RAYTRACINGMATH_H
#define RAYTRACINGMATH_H

#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include "macrosim_types.h"
#include <vector_types.h>
#include <optix_math.h>
#include "Geometry_Intersect.h"
#include "GlobalConstants.h"
#include "randomGenerator.h"

/**
 *  \brief Intersect ray with CCW wound triangle.  Returns non-normalize normal vector.
 */ 
inline RT_HOSTDEVICE bool my_intersect_triangle( float3 rayDirection,
											  float3 rayPosition,
                                               const float3& p0,
                                               const float3& p1,
                                               const float3& p2,
                                               float3& n,
                                               float&  t,
                                               float&  beta,
                                               float&  gamma )
{
  float3 e0 = p1 - p0;
  float3 e1 = p0 - p2;
  n  = cross( e0, e1 );

  float v   = dot( n, rayDirection );
  float r   = 1.0f / v;

  float3 e2 = p0 - rayPosition;
  float va  = dot( n, e2 );
  t         = r*va;

//  if(t < ray.tmax && t > ray.tmin) {
    float3 i   = cross( e2, rayDirection );
    float v1   = dot( i, e1 );
    beta = r*v1;
    if(beta >= 0.0f){
      float v2 = dot( i, e0 );
      gamma = r*v2;
      n = -n;
      return ( (v1+v2)*v <= v*v && gamma >= 0.0f );
    }
//  }
  return false;
}

/* rotate ray around given axis and do it in place */
inline RT_HOSTDEVICE void rotateRay(double3 &ray, double3 axis, double tilt)
{
	double3 rayTmp;
	double3 rotKrnlTmp;
	/* see http://de.wikipedia.org/wiki/Drehmatrix for reference of the rotation matrix used here */
	/* first component */
	rotKrnlTmp=make_double3(cos(tilt)+pow(axis.x,2)*(1-cos(tilt)),axis.x*axis.y*(1-cos(tilt))-axis.z*sin(tilt),axis.x*axis.z*(1-cos(tilt))+axis.y*sin(tilt));
	rayTmp.x=dot(ray, rotKrnlTmp);
	/* second component */
	rotKrnlTmp=make_double3(axis.y*axis.x*(1-cos(tilt))+axis.z*sin(tilt),cos(tilt)+pow(axis.y,2)*(1-cos(tilt)),axis.y*axis.z*(1-cos(tilt))-axis.x*sin(tilt));
	rayTmp.y=dot(ray, rotKrnlTmp);
	/* third component */
	rotKrnlTmp=make_double3(axis.z*axis.x*(1-cos(tilt))-axis.y*sin(tilt),axis.z*axis.y*(1-cos(tilt))+axis.x*sin(tilt), cos(tilt)+pow(axis.z,2)*(1-cos(tilt)));
	rayTmp.z=dot(ray, rotKrnlTmp);
	ray=rayTmp;
}

/* rotate ray around each coordinate axis and do it in place */
inline RT_HOSTDEVICE void rotateRay(double3 *ray, double3 tilt)
{
	double3 rayTmp;
	double3x3 Mx=make_double3x3(1,0,0, 0,cos(tilt.x),-sin(tilt.x), 0,sin(tilt.x),cos(tilt.x));
	double3x3 My=make_double3x3(cos(tilt.y),0,sin(tilt.y), 0,1,0, -sin(tilt.y),0,cos(tilt.y));
	double3x3 Mz=make_double3x3(cos(tilt.z),-sin(tilt.z),0, sin(tilt.z),cos(tilt.z),0, 0,0,1);
	double3x3 Mxy=Mx*My;
	double3x3 M=Mxy*Mz;
	rayTmp=M*(*ray);
	*ray=rayTmp;

	//double3 rayTmp;
	//double3 rotKrnlTmp;
	///* rotate phiX around x-axis */
	//double3x3 Mx=make_double3x3(1,0,0, 0,cos(tilt.x),-sin(tilt.x), 0,sin(tilt.x),cos(tilt.x));
	//rayTmp=Mx*(*ray);
	//*ray=rayTmp;
	///* rotate y- and z axis accordingly */
	//double3 eyPrime=Mx*make_double3(0,1,0);
	//double3 ezPrime=Mx*make_double3(0,0,1);
	///* rotate phiY around transformed y-axis */
	//rayTmp=eyPrime*dot(eyPrime,*ray)+cross(cos(tilt.y)*cross(eyPrime,*ray),eyPrime)+sin(tilt.y)*cross(eyPrime,*ray);
	//*ray=rayTmp;
	///* rotate z axis accordingly */
	//ezPrime=Mx*ezPrime;
	///* rotate around transformed z-axis */
	//rayTmp=ezPrime*dot(ezPrime,*ray)+cross(cos(tilt.z)*cross(ezPrime,*ray),ezPrime)+sin(tilt.z)*cross(ezPrime,*ray);
	//*ray=rayTmp;
}

/* rotate ray around each coordinate axis and do it in place */
inline RT_HOSTDEVICE void rotateRayInv(double3 *ray, double3 tilt)
{
	double3 rayTmp;
	tilt=-tilt;
	double3x3 Mx=make_double3x3(1,0,0, 0,cos(tilt.x),-sin(tilt.x), 0,sin(tilt.x),cos(tilt.x));
	double3x3 My=make_double3x3(cos(tilt.y),0,sin(tilt.y), 0,1,0, -sin(tilt.y),0,cos(tilt.y));
	double3x3 Mz=make_double3x3(cos(tilt.z),-sin(tilt.z),0, sin(tilt.z),cos(tilt.z),0, 0,0,1);
	double3x3 Mzy=Mz*My;
	double3x3 M=Mzy*Mx;
	rayTmp=M*(*ray);
	*ray=rayTmp;
}

/* rotate ray around each coordinate axis */
inline RT_HOSTDEVICE double3 rotateRay(double3 ray, double3 tilt)
{
	double3 rayTmp;
	double3 rotKrnlTmp;
	/* rotate phiX around x-axis */
	double3x3 Mx=make_double3x3(1,0,0, 0,cos(tilt.x),-sin(tilt.x), 0,sin(tilt.x),cos(tilt.x));
	rayTmp=Mx*(ray);
	ray=rayTmp;
	/* rotate y- and z axis accordingly */
	double3 eyPrime=Mx*make_double3(0,1,0);
	double3 ezPrime=Mx*make_double3(0,0,1);
	/* rotate phiY around transformed y-axis */
	rayTmp=eyPrime*dot(eyPrime,ray)+cross(cos(tilt.y)*cross(eyPrime,ray),eyPrime)+sin(tilt.y)*cross(eyPrime,ray);
	ray=rayTmp;
	/* rotate z axis accordingly */
	ezPrime=Mx*ezPrime;
	/* rotate around transformed z-axis */
	rayTmp=ezPrime*dot(ezPrime,ray)+cross(cos(tilt.z)*cross(ezPrime,ray),ezPrime)+sin(tilt.z)*cross(ezPrime,ray);
	ray=rayTmp;
	return ray;
}

/*intersectRayPlane is also in PlaneSurface_intersect.h. there it has a different list of arguments*/
inline RT_HOSTDEVICE double intersectRayPlane(double3 rayPosition, double3 rayDirection, double3 root, double3 normal)
{
	double tHelp1 = dot((rayPosition-root), normal);
	double tHelp2 = dot(rayDirection, normal); 
	if (tHelp2==0)
		return 0;
	return -tHelp1/tHelp2;
}

inline RT_HOSTDEVICE double calcDistRayPoint(double3 rayPosition, double3 rayDirection, double3 point)
{
  double3 pos2point=point-rayPosition;
  double numerator=length(cross(pos2point, rayDirection));
  return numerator/dot(rayDirection, rayDirection);
}

// return true if the ray hits inside the aperture. return false if not
inline RT_HOSTDEVICE bool checkAperture(double3 apertureCentre, double3 apertureTilt, double3 intersection, ApertureType type, double2 apertureHalfWidth)
{
	  //double2 test=apertureHalfWidth;
	  // an aperture half width of zero is interpreted as an infinite aperture. So we don't need to check and return true instantly !!
	  if ( (apertureHalfWidth.x==0) || (type==AT_INFTY) )
	  {
		  return true;
	  }

	  double3 normal=make_double3(0,0,1);
	  rotateRay(&normal,apertureTilt);
	  double3 root=apertureCentre;
	  //we calculate the vector from the intersection point to the ray through the aperture center with raydirection=apertureOrientation.
	  //therefor we define a plane through intersectionpoint parallel to aperture.
	  //we calculate intersection point of this plane with the ray through aperture center with direction=apertureOrientation
	  double3 tempPointOnNormal= intersectRayPlane(root,normal,intersection,normal)*normal+root;
	  //then the vector is defined as "intersectionpoint - point on axis of aperture"
	  double3 aprtCentre2Inters=intersection-tempPointOnNormal;
 
//	  double3 testPos=intersection;

//	  double2 aptTest=apertureHalfWidth;
	  
	  double3 rotatedVec=aprtCentre2Inters;
	  rotateRayInv(&rotatedVec, apertureTilt);
	  switch(type)
	  {
	  case AT_RECTOBSC:
		  // not implmented yet. Therefore we jump to AT_RECT and ignore the obscuration for now...
	  case AT_RECT:
		  if((abs(rotatedVec.x)<=apertureHalfWidth.x)&&(abs(rotatedVec.y)<=apertureHalfWidth.y))
			return true;
		  break;
	  case AT_ELLIPTOBSC:
		  // not implemented yet. Therefore we jump to AT_ELLIPT and ignore the obscuration for now...
	  case AT_ELLIPT:
		if ( pow(rotatedVec.x,2)/pow(apertureHalfWidth.x,2)+pow(rotatedVec.y,2)/pow(apertureHalfWidth.y,2) <= 1 )
			return true;
		break;

	  default:
		  // if we don't know the type of the aperture, we assume the ray doesn't hit the aperture
		  //return true;
		  break;
	  }
	  return false;
}


inline RT_HOSTDEVICE bool checkApertureCylinder(double3 root, double3 orientation, double3 intersection, double apertureHalfWidth)
{
	//to check the aperture we calculate the vector from the root to the intersection
	//we name it offsetVec
	//we calculate the part of the offsetVec that is parallel to the orientation of the cylinder/cone
	//if this vector is longer than apertureHalfwidth, the intersection is in the invalid area.

	//if we later want to implement the possibility to check an aperture not only for the length, then we can calculate the orthogonal offsetvec as: ortVec=offsetVec-parallelVec
	
	double3 offsetVec=intersection-(root+apertureHalfWidth*orientation);
	double3 offsetVecParallel=dot(offsetVec,orientation)*orientation;
	if (length(offsetVecParallel)<=apertureHalfWidth)
	{
		return true;
	}
	return false;
}

inline RT_HOSTDEVICE bool calcSnellsLaw(double3 *rayDirection, double3 interfaceNormal, double ni, double no)
{
 // /* see Michael J. Kidger, Fundamental Optical Design, pp54 */
 // double q=ni/no;
 // double cosPhiIn=dot(*rayDirection, interfaceNormal);
 // double cosPhiOut=sqrt(1-q*q*(1-cosPhiIn*cosPhiIn));
 // /* check for total internal reflection */
 // if ( q*q*(1-cosPhiIn*cosPhiIn) > 1 )
 // {
	///* if we have total internal reflection calculate reflection */
	//	*rayDirection=(*rayDirection-2*cosPhiIn*interfaceNormal);
	//return false;
 // }
 // else
 // {
	//  /* calc the refraction */
	//  // we need to distinguish wether the surface normal is pointing in the same direction as the ray or not...
	//if (cosPhiIn>0)
	//	*rayDirection=(q*(*rayDirection)+interfaceNormal*(cosPhiOut-q*cosPhiIn));
	//else
	//	*rayDirection=(q*(*rayDirection)-interfaceNormal*(cosPhiOut-q*cosPhiIn));
	//return true;
 // }
 
  //see Dissertation Stolz: "Differentielles Raytracing für spezielle Beleuchtungssysteme", pp. 20
 double mu=ni/no;
 double s=dot(*rayDirection,interfaceNormal);
 double testSign=copy_sign( (double)1.0, s );
 double test=sqrt(1-mu*mu*(1-s*s));

 if (mu*mu*(1-s*s)>1)
 {
	  //we have total internal reflection
	 *rayDirection=reflect(*rayDirection,interfaceNormal);
	 return false; // signal that no refraction was taking place. Therefore the immersion of the ray will not be changed
 }
 double gamma_r=-mu*s+copy_sign( (double)1.0, s )*sqrt(1-mu*mu*(1-s*s));
 *rayDirection=mu*(*rayDirection)+gamma_r*interfaceNormal;
 return true;
}

/* create transformation matrix from rotation and translation */
inline RT_HOSTDEVICE double4x4 createTransformationMatrix(double3 rotation, double3 translation)
{

	double4x4 t;
	double3x3 rotMatX=make_double3x3(1,0,0, 0,cos(rotation.x),-sin(rotation.x), 0,sin(rotation.x),cos(rotation.x));
	double3x3 rotMatY=make_double3x3(cos(rotation.y),0,sin(rotation.y), 0,1,0, -sin(rotation.y),0,cos(rotation.y));
	double3x3 rotMatZ=make_double3x3(cos(rotation.z),-sin(rotation.z),0, sin(rotation.z),cos(rotation.z),0, 0,0,1);
	double3x3 rotMat=rotMatX*rotMatY;
	rotMat=rotMat*rotMatZ;

	t=make_double4x4(rotMat.m11,rotMat.m12,rotMat.m13,translation.x, rotMat.m21,rotMat.m22,rotMat.m23,translation.y, rotMat.m31,rotMat.m32,rotMat.m33,translation.z, 0,0,0,1);
	return t;
}

inline RT_HOSTDEVICE void transformDifferentialData(const double3 N, const double3 P, const double2 radius, const double3 NBar, double3 &P_r, double3 &T_r, double2 &radius_r, double &torsion_r )
{
	// see Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010 (eq.3.17 on page 23)
	P_r=cross(N,NBar);
	if (length(P_r)==0)
		P_r=P; // if direction and normal are parallel, we can arbitrarily chose one of the mainDirections as Pr. ( can we really??? )
	P_r=normalize(P_r);
	T_r=normalize(cross(N,P_r));
	double cosPhi=dot(P_r, P);
	double sinPhi=-dot(T_r, P);
	double2 curv_r;
	curv_r.x=cosPhi*cosPhi/radius.x+sinPhi*sinPhi/radius.y;

	curv_r.y=sinPhi*sinPhi/radius.x+cosPhi*cosPhi/radius.y;
	radius_r.x=1/curv_r.x;
	radius_r.y=1/curv_r.y;
	torsion_r=(1/radius.x-1/radius.y)*sinPhi*cosPhi; // note that torsion equals 1/sigma_r in Stolz's dissertation
}

inline RT_HOSTDEVICE void invTransformDifferentialData(const double3 P_r, const double3 T_r, const double2 radius_r, const double torsion_r, double3 &P, double3 &T, double2 &radius)
{
	// see Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010 (eq.3.18 on page 23)
	double Phi;
	if (torsion_r==0)
		Phi=0;
	else
	{
		if (radius_r.x==radius_r.y)
			Phi=torsion_r/abs(torsion_r)*PI/2/2; //atan(+-inf)=+-pi/2 ...
		else
			Phi=(atan(2*torsion_r/(1/radius_r.x-1/radius_r.y))/2);
	}
	double cosPhi=cos(Phi);
	double sinPhi=sin(Phi);
	double2 curv;
	curv.x=cosPhi*cosPhi/radius_r.x+sinPhi*sinPhi/radius_r.y-sin(2*Phi)*torsion_r;
	curv.y=sinPhi*sinPhi/radius_r.x+cosPhi*cosPhi/radius_r.y+sin(2*Phi)*torsion_r;
	radius.x=1/curv.x;
	radius.y=1/curv.y;
	P=cosPhi*P_r-sinPhi*T_r;
	T=sinPhi*P_r+cosPhi*T_r;
}

inline RT_HOSTDEVICE void refractDifferentialData(double3 N, double3 NBar, double3 NPrime,  double3 P_r, double2 radius_r, double2 radiusBar_r, double torsion_r, double torsionBar_r, double mu, double3 &PPrime_r, double3 &TPrime_r, double2 &radiusPrime_r, double &torsionPrime_r, double &flux)
{
	// see Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010 (eq.3.19 on page 24)
	double s=dot(N,NBar);
	int signS=(int)(s/abs(s));
	PPrime_r=P_r;
	TPrime_r=normalize(cross(N,PPrime_r));
	double gamma=-mu*s+signS*sqrt(1-mu*mu*(1-s*s));
	double2 curvPrime_r;
	curvPrime_r.x=mu/radius_r.x+gamma/radiusBar_r.x;
	double sPrime=dot(N,NBar);
	curvPrime_r.y=1/(sPrime*sPrime)*(mu*s*s/radius_r.y+gamma/radiusBar_r.y);
	radiusPrime_r.x=1/curvPrime_r.x;
	radiusPrime_r.y=1/curvPrime_r.y;
	torsionPrime_r=1/sPrime*(mu*s*torsion_r+gamma*torsionBar_r);
	flux=flux*abs(dot(N,NBar)/dot(NPrime,NBar));
}

inline RT_HOSTDEVICE void reflectDifferentialData(double3 N, double3 NBar, double3 NPrime,  double3 P_r, double2 radius_r, double2 radiusBar_r, double torsion_r, double torsionBar_r, double mu, double3 &PPrime_r, double3 &TPrime_r, double2 &radiusPrime_r, double &torsionPrime_r, double &flux)
{
	// see Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010 (eq.3.19 on page 24)
	double s=dot(N,NBar);
	int signS=(int)(s/abs(s));
	PPrime_r=P_r;
	TPrime_r=normalize(cross(N,PPrime_r));
	double gamma=-2*s;
	double2 curvPrime_r;
	curvPrime_r.x=mu/radius_r.x+gamma/radiusBar_r.x;
	double sPrime=dot(N,NBar);
	curvPrime_r.y=1/(sPrime*sPrime)*(mu*s*s/radius_r.y+gamma/radiusBar_r.y);
	radiusPrime_r.x=1/curvPrime_r.x;
	radiusPrime_r.y=1/curvPrime_r.y;
	torsionPrime_r=1/sPrime*(mu*s*torsion_r+gamma*torsionBar_r);
	flux=flux*abs(dot(N,NBar)/dot(NPrime,NBar));
}

// calc the angle of a vector with respect to a coordinate system that is rotated by tilt with respect to the global coordinate system
inline RT_HOSTDEVICE double2 calcAnglesFromVector(const double3 &vec, const double3 tilt)
{
	// calc local coordinate axis
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);

	rotateRay(&t_ez,tilt);
	rotateRay(&t_ey,tilt);
	rotateRay(&t_ex,tilt);
	
	// we assume that we are interested in angles with respect to x- and y-axis
	// calc projection of ray onto local x-axis
	double t_x=dot(t_ex,vec);
	// remove x-component from ray
	double3 t_ray_y=normalize(vec-t_x*t_ex);
	// calc rotation angle around x with respect to z axis
	double phi_x =acos(dot(t_ray_y,t_ez));
	// calc projection of ray onto local y-axis
	double t_y=dot(t_ey,vec);
	// in order to get the sign right we need to check the sign of the projection on th y-axis
	if (t_y>0)
		phi_x=-phi_x;
	// remove y-component from ray
	double3 t_ray_x=normalize(vec-t_y*t_ey);
	// calc rotation angle around y with respect to z axis
	double phi_y=acos(dot(t_ray_x,t_ez));
	// in order to get the sign right we need to check the sign of the projection on th y-axis
	if (t_x>0)
		phi_y=-phi_y;	

	return make_double2(phi_x,phi_y);
}

// distribute ray direction uniformly to hit importance area
inline RT_HOSTDEVICE void aimRayTowardsImpArea(double3 &direction, double3 position, double3 impAreaRoot,  double2 impAreaHalfWidth, double3 impAreaTilt, ApertureType impAreaType, unsigned int &currentSeed)
{

	uint32_t x1[5]; // variable for random generator			
	
	// declar variables for randomly distributing ray directions via an importance area
//	double3 dirImpAreaCentre, tmpPos, impAreaRoot;
	double3 tmpPos;

	RandomInit(currentSeed, x1); // init random variable

	double impAreaX;
	double impAreaY;
			
	// now distribute points inside importance area

	if (impAreaType==AT_RECT)
	{
		// place temporal point uniformingly randomly inside the importance area
		impAreaX=(Random(x1)-0.5)*2*impAreaHalfWidth.x;
		impAreaY=(Random(x1)-0.5)*2*impAreaHalfWidth.y; 
	}
	else 
	{
		if (impAreaType==AT_ELLIPT)
		{
			double theta=2*PI*Random(x1);
			double r=sqrt(Random(x1));
			impAreaX=impAreaHalfWidth.x*r*cos(theta);
			impAreaY=impAreaHalfWidth.y*r*sin(theta);
		}
	}
		
	
	double3 impAreaAxisX=make_double3(1,0,0);
	double3 impAreaAxisY=make_double3(0,1,0);
		
	rotateRay(&impAreaAxisX,impAreaTilt);
	rotateRay(&impAreaAxisY,impAreaTilt);

	tmpPos=impAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
	direction=normalize(tmpPos-position);


	//uint32_t x1[5];
	//RandomInit(currentSeed, x1); // init random variable

	//double3 exApt=make_double3(1,0,0);
	//double3 eyApt=make_double3(0,1,0);

	//rotateRay(&exApt,impAreaTilt);
	//rotateRay(&eyApt,impAreaTilt);

	//double x, y;
	//if (impAreaType==AT_RECT)
	//{
	//	x=(Random(x1)*2-1)*impAreaHalfWidth.x;
	//	y=(Random(x1)*2-1)*impAreaHalfWidth.y;
	//}
	//else // default is elliptical
	//{
	//	double theta=2*PI*Random(x1);
	//	double r=sqrt(Random(x1));
	//	x=impAreaHalfWidth.x/2*r*cos(theta);
	//	y=impAreaHalfWidth.y/2*r*sin(theta);
	//}
	//double3 tmpPos=impAreaRoot+x*exApt+y*eyApt;
	//direction=normalize(tmpPos-position);


	// this distributes direction of ray normally. 
	// But it only works if the importance area is located centrally in front of the ray position

	//// calc opening angle into importance area
	//double alphaX_Opening=acosf(dot(normalize(impAreaRoot+exApt*impAreaHalfWidth.x),normalize(impAreaRoot-exApt*impAreaHalfWidth.x)))/2;
	//double alphaY_Opening=acosf(dot(normalize(impAreaRoot+eyApt*impAreaHalfWidth.y),normalize(impAreaRoot-eyApt*impAreaHalfWidth.y)))/2;
	////double alpha0=acosf(dot(normalize(impAreaRoot),normalize(position)));
	//double alphaX, alphaY;

	//if (impAreaType==AT_RECT)
	//{
	//	alphaX=(Random(x1)*2-1)*alphaX_Opening;
	//	alphaY=(Random(x1)*2-1)*alphaY_Opening;
	//}
	//else // default is elliptical
	//{
	//	double theta=2*PI*Random(x1);
	//	double r=sqrt(Random(x1));
	//	alphaX=alphaX_Opening/2*r*cos(theta);
	//	alphaY=alphaY_Opening/2*r*sin(theta);
	//}

	//double x,y,z;
	//if ( (alphaX==0) && (alphaY==0) )
	//{
	//	x=0;
	//	y=0;
	//	z=1;
	//}
	//else
	//{
	//	if (abs(alphaX)>abs(alphaY))
	//	{
	//		x=sqrt(1/(1+pow(tan(alphaY),2)/pow(tan(alphaX),2)+1/pow(tan(alphaX),2)));
	//		if (alphaX<0)
	//			x=-x;
	//		z=x/tan(alphaX);
	//		y=tan(alphaY)*z;
	//	}
	//	else
	//	{
	//		y=sqrt(1/(1+pow(tan(alphaX),2)/pow(tan(alphaY),2)+1/pow(tan(alphaY),2)));
	//		if (alphaY<0)
	//			y=-y;
	//		z=y/tan(alphaY);
	//		x=tan(alphaX)*z;
	//	}
	//}
	//direction=make_double3(x,y,z);
	//rotateRay(&direction,impAreaTilt);

	// this distributes the ray direction into the cone of  the importance area
	// but it doesn't distribute it normally for some reason
	// init direction towards centre of importance area
	//double3 test1=impAreaRoot;
	//double3 test2=position;
	//direction=normalize(impAreaRoot-position);
	//// calc opening angle into importance area
	//double alphaX_Opening=acosf(dot(normalize(impAreaRoot+exApt*impAreaHalfWidth.x),normalize(impAreaRoot-exApt*impAreaHalfWidth.x)))/2;
	//double alphaY_Opening=acosf(dot(normalize(impAreaRoot+eyApt*impAreaHalfWidth.y),normalize(impAreaRoot-eyApt*impAreaHalfWidth.y)))/2;
	////double alpha0=acosf(dot(normalize(impAreaRoot),normalize(position)));
	//double alphaX, alphaY;

	//if (impAreaType==AT_RECT)
	//{
	//	alphaX=(Random(x1)*2-1)*alphaX_Opening;
	//	alphaY=(Random(x1)*2-1)*alphaY_Opening;
	//}
	//else // default is elliptical
	//{
	//	double theta=2*PI*Random(x1);
	//	double r=sqrt(Random(x1));
	//	alphaX=alphaX_Opening/2*r*cos(theta);
	//	alphaY=alphaY_Opening/2*r*sin(theta);
	//}
	//rotateRay(&direction,make_double3(alphaX,alphaY,0));

	currentSeed=x1[4]; // save new seed for next randomization
};


// note that refVec has to be normal to rotAxis in order to get meaningful results here
inline RT_HOSTDEVICE void calcRotationAroundAxis(const double3 &vec, const double3 &refVec, const double3 &rotAxis, double &phi)
{
	// calc projection of ray onto rotation axis
	double t_x=dot(rotAxis,vec);
	// remove x-component from ray
	double3 t_vec_y=normalize(vec-t_x*rotAxis);
	// calc rotation angle around x with respect to reference vector
	phi =acos(dot(t_vec_y, refVec));
};

// creates a unit vector whose projection on the xz-plane intersects the yz-plane at an angle phi.y.
// similarly the projection of the created vec onto the yz plane will intersect the xz plane at an angle phi.x
inline RT_HOSTDEVICE double3 createObliqueVec(const double2 &phi)
{
	return normalize(make_double3(tan(phi.y),tan(phi.x),1));
};


// functions for 2D bicubic spline interpolation


#endif