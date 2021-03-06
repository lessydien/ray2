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

/**\file Scatter_Phong_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SCATTER_PHONG_HIT_H
  #define SCATTER_PHONG_HIT_H
  
#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "rayData.h"
#include "Scatter_hit.h"


#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   ScatPhong_params
  *\ingroup Scatter
  *\brief   reduced set of params that are loaded onto the GPU if the tracing is done there
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
class ScatPhong_params: public Scatter_ReducedParams
{
public:
	double coefLambertian; // coef reflectance for lambertian
	double phongParam; // factor n for cos
	double coefPhong; // factor for Phong
};

/**
 * \detail hitPhong 
 *
 * modifies the raydata according to the parameters of the scatter
 *
 * \param[in] rayStruct &prd, Mat_hitParams hitParams, ScatPhong_params params
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitPhong(rayStruct &prd, Mat_hitParams hitParams, ScatPhong_params params)
{
	uint32_t x1[5];
	RandomInit(prd.currentSeed, x1); // init random variable

	prd.running=true;

	ScatPhong_params test=params;
//	params.phongParam;   // cos(a)^n,  the parameter ne
//	params.coefPhong;    // reflectance of cos(a)^n
//	params.coefLambertian;
	// adjust flux of ray according to TIR of surface
	//prd.flux=prd.flux*sqrt(params.TIR);  // here TIR is the reflectance of lambertian
	double3 incidentDirection=normalize(reflect(prd.direction,-hitParams.normal));
    double3 mirrorReflectionDirection=prd.direction;
	//double3 reflectionDirection=reflect(prd.direction,hitParams.normal);
	double intensityCosLambertian=dot(hitParams.normal,prd.direction)/sqrt(dot(hitParams.normal,hitParams.normal)*dot(prd.direction,prd.direction));
	//intensityCosLambertian=abs(intensityCosLambertian);
	//double intensityCosLambertian=dot(hitParams.normal,prd.direction)/abs(dot(hitParams.normal,prd.direction));
	//if (intensityCosLambertian<0){intensityCosLambertian=0;}
	// if we had no importance area in prescription file, the parser set one that corresponds to the full hemisphere...
	// if we have no importance area, we scatter into full hemisphere. Directions are distributed according to BRDF and flux is constant
	if (params.impAreaType==AT_INFTY)
	{
		//calc scatter angle in x. directions are uniformly distributed
		double xix=(Random
			(x1)-0.5)*PI;
		double xiy=(Random(x1)-0.5)*PI;
		// calculate outcome direction
		
		//double3 incidentDirection=prd.direction;
		// normalization of surface vector
		//hitParams.normal=hitParams.normal/sqrt(dot(hitParams.normal,hitParams.normal));
		// mirror reflection vector
		//double3 mirrorReflectionDirection=incidentDirection-2*dot(incidentDirection,hitParams.normal)*hitParams.normal;
		
			//=reflect(prd.direction,hitParams.normal);
		
		//adjust for lambertian cos intensity lose due to incident angle
		
		// scattered ray direction is independent of incoming direction, so possibility of random reflected ray should be hemisphere based on hitsphere
		// not based on mirrorreflected ray.
		prd.direction=hitParams.normal;    
		// scattered ray direction is independent of incoming direction-> Init to normal of surface. consider reflection ...
		//prd.direction=intensityCosLambertian*hitParams.normal;
		//prd.direction=dot(hitParams.normal,prd.direction)/abs(dot(hitParams.normal,prd.direction))*hitParams.normal;

		
		
		// rotate according to scattering angles in x and y
		rotateRay(&prd.direction, make_double3(xix, xiy, 0));

		double cosTheta=dot(prd.direction,mirrorReflectionDirection)/sqrt(dot(prd.direction,prd.direction)*dot(mirrorReflectionDirection,mirrorReflectionDirection));
		//double cosTheta=dot(prd.direction,mirrorReflectionDirection)/abs(dot(prd.direction,mirrorReflectionDirection));
		if (intensityCosLambertian<0 && intensityCosLambertian>1){
			intensityCosLambertian=0;
		}
		
		if (cosTheta<0){
			cosTheta=0;
		}
		//prd.flux=intensityCosLambertian*prd.flux*sqrt(params.coefLambertian)+sqrt(params.coefPhong)*prd.flux*pow(cosTheta,params.phongParam);
		prd.flux=intensityCosLambertian*(prd.flux*params.coefLambertian+params.coefPhong*prd.flux*pow(cosTheta,params.phongParam));
	}
	else // if we have an importance area, we scatter into this area. Directions are uniformly distributed and flux adjusted according to BRDF
	{
        aimRayTowardsImpArea(prd.direction, prd.position, params.impAreaRoot,  params.impAreaHalfWidth, params.impAreaTilt, params.impAreaType, prd.currentSeed);

		//double3 impAreaNormal=make_double3(0,0,1);  
		//rotateRay(&impAreaNormal, params.impAreaTilt);  // transform normal vetor of new surface
		//double distance_twosurfaces=sqrt(pow((prd.position.x-params.impAreaRoot.x),2)+pow((prd.position.y-params.impAreaRoot.y),2)+pow((prd.position.z-params.impAreaRoot.z),2));
		//double3 direction_light=params.impAreaRoot-prd.position;
		//double hitParamslength=sqrt(hitParams.normal.x*hitParams.normal.x+hitParams.normal.y+hitParams.normal.y+hitParams.normal.z*hitParams.normal.z);
		//double prdNormallength=sqrt(impAreaNormal.x*impAreaNormal.x+impAreaNormal.y*impAreaNormal.y+impAreaNormal.z*impAreaNormal.z);
		//double theta1=(dot(hitParams.normal,direction_light)/hitParamslength/distance_twosurfaces);
		//double theta2=(dot(impAreaNormal,direction_light)/prdNormallength/distance_twosurfaces);
		

		double cosTheta=dot(prd.direction,mirrorReflectionDirection)/sqrt(dot(prd.direction,prd.direction)*dot(mirrorReflectionDirection,mirrorReflectionDirection));
		//double cosTheta=dot(prd.direction,mirrorReflectionDirection)/abs(dot(prd.direction,mirrorReflectionDirection));
		if (intensityCosLambertian<0 && intensityCosLambertian>1){
			intensityCosLambertian=0;
		}
		
		if (cosTheta<0){
			cosTheta=0;
		}
		prd.flux=intensityCosLambertian*(prd.flux*params.coefLambertian+params.coefPhong*prd.flux*pow(cosTheta,params.phongParam));
	
		//rotateRay(&impAreaX,params.impAreaTilt);
		//double3 impAreaY=make_double3(0,1,0);
		//rotateRay(&impAreaY,params.impAreaTilt);
		//// calc opening angle into importance area
		//double alphaX=acos(dot(normalize(params.impAreaRoot+impAreaX*params.impAreaHalfWidth.x),normalize(params.impAreaRoot-impAreaX*params.impAreaHalfWidth.x)))/2;
		//double alphaY=acos(dot(normalize(params.impAreaRoot+impAreaY*params.impAreaHalfWidth.y),normalize(params.impAreaRoot-impAreaY*params.impAreaHalfWidth.y)))/2;
		//// normalize flux according to solid angle of importance area
		//double omegaS;
		//double3 da;
		//double rho;
		//if (params.impAreaType==AT_RECT)
		//{
		//	da=make_double3(0,0,1);
		//	rotateRay(&da,params.impAreaTilt);
		//	da=da*params.impAreaHalfWidth.x*params.impAreaHalfWidth.y;
		//	rho=length(params.impAreaRoot-prd.position);
		//	rho=rho*rho;
		//	omegaS=dot(prd.direction,da)/rho;
		//	prd.flux=omegaS/4*PI; 
		//}
		//if (params.impAreaType==AT_ELLIPT)
		//{
		//	da=make_double3(0,0,1);
		//	rotateRay(&da,params.impAreaTilt);
		//	da=da*PI*params.impAreaHalfWidth.x*params.impAreaHalfWidth.y;
  //          rho=length(params.impAreaRoot-prd.position);
		//	rho=rho*rho;
		//	omegaS=dot(prd.direction,da)/rho;
		//	prd.flux=omegaS/4*PI; 
		//}

//		double3 impAreaNormal=make_double3(0,0,1);
//		rotateRay(&impAreaNormal,params.impAreaTilt);
//		prd.flux=prd.flux*abs(dot(hitParams.normal,prd.direction)); // apply lambert intensity distribution

	}
    RandomInit(prd.currentSeed, x1); // init random variable
    prd.currentSeed=x1[4];

	return true;
};


#endif
