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

#ifndef SCATTER_COOKTORRANCE_HIT_H
  #define SCATTER_COOKTORRANCE_HIT_H
  
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
class ScatCookTorrance_params: public Scatter_ReducedParams
{
public:
	double coefLambertian; // coef reflectance for lambertian
	double fresnelParam; // factor n for cos
	double roughnessFactor; // factor for Phong
};


inline RT_HOSTDEVICE bool hitCookTorrance(rayStruct &prd, Mat_hitParams hitParams, ScatCookTorrance_params params)
{
	uint32_t x1[5];
	RandomInit(prd.currentSeed, x1); // init random variable

	prd.running=true;

	ScatCookTorrance_params test=params;

    double3 mirrorReflectionDirection=prd.direction;
	double3 incidentDirection=normalize(reflect(prd.direction,-hitParams.normal));
	
	double NdotL=max((dot(hitParams.normal,prd.direction)/sqrt(dot(hitParams.normal,hitParams.normal)*dot(prd.direction,prd.direction))),0.0);
	//if (NdotL<=0){
	//	NdotL=0;}
	double intensityCosLambertian=NdotL;

	//double intensityCosLambertian=abs(dot(hitParams.normal,-incidentDirection));

	double specular = 0.0;		
	hitParams.normal=normalize(hitParams.normal);

	if (params.impAreaType==AT_INFTY)
	{
		//calc scatter angle in x. directions are uniformly distributed
		double xix=(Random
			(x1)-0.5)*PI;
	    double xiy=(Random(x1)-0.5)*PI;

		prd.direction=hitParams.normal;
		
		//prd.direction=dot(hitParams.normal,prd.direction)/abs(dot(hitParams.normal,prd.direction))*hitParams.normal
		if (NdotL>0.0){
		
		// rotate according to scattering angles in x and y
		
		prd.direction=normalize(prd.direction);
		rotateRay(&prd.direction, make_double3(xix, xiy, 0));
		// recalucate normal h, prd.direction is n
		double3 newNormal=normalize(prd.direction-incidentDirection);
		double NdotV=max(dot(hitParams.normal,prd.direction),0.0);
		
		double VdotH=max(dot(prd.direction,newNormal),0.0);
		double NdotH=max(dot(hitParams.normal,newNormal),0.0);
		double mSquared=pow(params.roughnessFactor,2.0);
		//double FresnelcosTheta=dot(newNormal,prd.direction)/abs(dot(newNormal,prd.direction));

		//Fresnel schlick approximation
		double reflectionFresnel=params.fresnelParam+(1-params.fresnelParam)*pow((1-VdotH),5);


		//Geometric Attenuation Factor
		double Gmasking=2.0*NdotH*NdotV/VdotH;
		double Gshadowing=2.0*NdotH*NdotL/VdotH;
		double Gfactor=min(1.0,min(Gmasking,Gshadowing));

		//Gfactor=min(Gfactor,1);


		//roughness factor of the surface
		double r1=1.0/(4.0*mSquared*pow(NdotH,4));
		double r2=(NdotH*NdotH-1.0)/(mSquared*NdotH*NdotH);
		double Dfactor=r1*exp(r2);
		//double Dfactor=1/PI/(pow(params.roughnessFactor,2)/pow((dot(hitParams.normal,newNormal)),4))*exp((pow((dot(hitParams.normal,newNormal)),2)-1)/(pow(params.roughnessFactor,2)/pow((dot(hitParams.normal,newNormal)),2)));
		specular=(reflectionFresnel*Gfactor*Dfactor)/PI/(NdotV*NdotL);
		}
		prd.flux=intensityCosLambertian*prd.flux*(params.coefLambertian+(1-params.coefLambertian)*specular);
		//prd.flux=intensityCosLambertian*prd.flux*sqrt(params.coefLambertian)+intensityCosLambertian*reflectionFresnel*Dfactor*Gfactor/dot(hitParams.normal,-incidentDirection)/dot(hitParams.normal,prd.direction);
	}
	else // if we have an importance area, we scatter into this area. Directions are uniformly distributed and flux adjusted according to BRDF
	{
        aimRayTowardsImpArea(prd.direction, prd.position, params.impAreaRoot,  params.impAreaHalfWidth, params.impAreaTilt, params.impAreaType, prd.currentSeed);
		// calculate outcome direction
		
		//double3 incidentDirection=prd.direction;
		// normalization of surface vector
		//hitParams.normal=hitParams.normal/sqrt(dot(hitParams.normal,hitParams.normal));
		// mirror reflection vector
		//double3 mirrorReflectionDirection=incidentDirection-2*dot(incidentDirection,hitParams.normal)*hitParams.normal;
		
			//=reflect(prd.direction,hitParams.normal);
		//prd.direction=hitParams.normal;
		//prd.direction=dot(hitParams.normal,prd.direction)/abs(dot(hitParams.normal,prd.direction))*hitParams.normal
		if (NdotL>0.0){
		
		// rotate according to scattering angles in x and y
		//rotateRay(&prd.direction, make_double3(xix, xiy, 0));
		prd.direction=normalize(prd.direction);
		// recalucate normal h, prd.direction is n
		double3 newNormal=normalize(prd.direction-incidentDirection);
		double NdotV=max(dot(hitParams.normal,prd.direction),0.0);
		
		double VdotH=max(dot(prd.direction,newNormal),0.0);
		double NdotH=max(dot(hitParams.normal,newNormal),0.0);
		double mSquared=pow(params.roughnessFactor,2.0);
		//double FresnelcosTheta=dot(newNormal,prd.direction)/abs(dot(newNormal,prd.direction));

		//Fresnel schlick approximation
		double reflectionFresnel=params.fresnelParam+(1-params.fresnelParam)*pow((1-VdotH),5);


		//Geometric Attenuation Factor
		double Gmasking=2.0*NdotH*NdotV/VdotH;
		double Gshadowing=2.0*NdotH*NdotL/VdotH;
		double Gfactor=min(1.0,min(Gmasking,Gshadowing));

		//Gfactor=min(Gfactor,1);


		//roughness factor of the surface
		double r1=1.0/(4.0*mSquared*pow(NdotH,4));
		double r2=(NdotH*NdotH-1.0)/(mSquared*NdotH*NdotH);
		double Dfactor=r1*exp(r2);
		//double Dfactor=1/PI/(pow(params.roughnessFactor,2)/pow((dot(hitParams.normal,newNormal)),4))*exp((pow((dot(hitParams.normal,newNormal)),2)-1)/(pow(params.roughnessFactor,2)/pow((dot(hitParams.normal,newNormal)),2)));
		specular=(reflectionFresnel*Gfactor*Dfactor)/PI/(NdotV*NdotL);
		}
		prd.flux=intensityCosLambertian*prd.flux*(params.coefLambertian+(1-params.coefLambertian)*specular);
		//prd.flux=intensityCosLambertian*prd.flux*sqrt(params.coefLambertian)+intensityCosLambertian*reflectionFresnel*Dfactor*Gfactor/dot(hitParams.normal,-incidentDirection)/dot(hitParams.normal,prd.direction);
	}
    RandomInit(prd.currentSeed, x1); // init random variable
    prd.currentSeed=x1[4];

	return true;
};


#endif
