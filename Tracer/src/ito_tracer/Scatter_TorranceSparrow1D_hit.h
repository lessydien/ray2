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

/**\file Scatter_TorranceSparrow1D_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SCATTER_TORRANCESPARROW1D_HIT_H
  #define SCATTER_TORRANCESPARROW1D_HIT_H
  
#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "rayData.h"
#include "Scatter_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   ScatTorranceSparrow1D_params
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
class ScatTorranceSparrow1D_params: public Scatter_ReducedParams
{
public:
	double Kdl; // coefficient of diffuse lobe
	double Ksl; // coefficient of specular lobe
	double Ksp; // coefficient of specular peak
	double sigmaXsl; // width parameter of specular lobe
	double sigmaXsp; // width parameter of specular peak
	double3 scatAxis;
};

/**
 * \detail hitTorranceSparrow1D 
 *
 * modifies the raydata according to the parameters of the scatter
 *
 * \param[in] rayStruct &prd, Mat_hitParams hitParams, ScatTorranceSparrow1D_params params
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitTorranceSparrow1D(rayStruct &prd, Mat_hitParams hitParams, ScatTorranceSparrow1D_params params)
{
	uint32_t x1[5];
	RandomInit(prd.currentSeed, x1); // init random variable
	//// project incoming ray onto plane containing geometry orientation and scattering axis
	double3 eInB=hitParams.normal*dot(prd.direction,hitParams.normal); // calc projection of incoming ray on normal
	double3 eInA=params.scatAxis*dot(prd.direction,params.scatAxis); // calc projection of incoming ray on scattering axis
	double3 eIn1=eInA+eInB; // part of direction ray in plane containing geometry normal and scattering axis
	double3 eIn2=prd.direction-eIn1; // part of direction ray in plane containing geometry normal and non scattering axis
	double angleRay2Normal=acos(dot(normalize(eIn1),-hitParams.normal)); // calc angle of ray to normal
	double3 rotAxis=cross(params.scatAxis,hitParams.normal); // calc axis around which these vectors are rotated to each other
	
	// calc maximum scatter angles
	double alphaMax;
	double alphaMin;
	if (params.impAreaType==AT_INFTY)
	{
		alphaMax=PI/2;
		alphaMin=-PI/2;
	}


	// calc scatter angle
	double xi=RandomGauss(x1); // get normal distribution with zero mean and unit variance
	double sqrtpi=sqrt(PI);
	double BRDFmax=params.Kdl*2+sqrtpi*(params.Ksl*params.Ksl+params.Ksp*params.Ksp);
	double ii=Random(x1)*BRDFmax; // get uniform distribution between zero and maximum BRDF to decide which mean and variance we need for our gaussian distribution
	double sigma=params.sigmaXsp; // init sigma to sigma of specular peak
	if (ii>sqrtpi*params.Ksp*params.Ksp)
	{
		prd.depth++;
		sigma=params.sigmaXsl; // set sigma to sigma of specular lobe
	}
	xi=xi*sigma;
	if (ii>sqrtpi*params.Ksp+params.Ksl*params.Ksl)
	{
		// the diffuse lobe distributes the direction uniformly
		xi=(Random(x1)-0.5)*PI; // get uniform deviate between -pi/2 pi/2
		//xi=asin(xi);// if we end up in the diffuse lobe we calculate xi from the inversion mehtod. see W.Press, Numerical Recipes 2nd ed., pp. 291 for reference
		prd.depth++;
	}
	prd.currentSeed=x1[4]; // save new seed for next randomization

	// rotate ray according to scattering angle
	rotateRay(eIn1, rotAxis, xi);
	prd.direction=eIn1+eIn2;
	return true;
};

//inline RT_HOSTDEVICE bool hitTorranceSparrow1D(rayStruct &prd, double3 hitParams.normal, ScatTorranceSparrow1D_params params)
//{
//	uint32_t x1[5];
//	RandomInit(prd.currentSeed, x1); // seed random generator
//	prd.currentSeed=x1[4]; // save seed for next round
//	// project incoming ray onto plane containing geometry orientation and scattering axis
//	double3 eInB=hitParams.normal*dot(prd.direction,hitParams.normal); // calc projection of incoming ray on normal
//	double3 eInA=params.scatAxis*dot(prd.direction,params.scatAxis); // calc projection of incoming ray on scattering axis
//	double3 eIn1=eInA+eInB; // part of direction ray in plane containing geometry normal and scattering axis
//	double3 eIn2=prd.direction-eIn1; // part of direction ray in plane containing geometry normal and non scattering axis
////	double3 eOut2=reflect(eIn2,hitParams.normal); // second partial ray is reflected conventionally
////	double3 eOut1=reflect(normalize(eIn1),hitParams.normal); // init eOut1 with specular reflection
////	double angleRay2Normal=acos(dot(normalize(eOut1),-hitParams.normal)); // calc angle of specular reflection to normal
//	double angleRay2Normal=acos(dot(normalize(eIn1),-hitParams.normal)); // calc angle of specular reflection to normal
//	// we're intereseted in reflection no matter wether we hit the front or back surface of the geometry. so the angle must be smaller than  pi/2
//	//if (abs(angleRay2Normal)>PI/2)
//	//	angleRay2Normal=acos(dot(normalize(eOut1),hitParams.normal)); // calc angle of specular reflection to normal
//	// here we need to determine the sign of the specular angle. Because I can't think of any smart way to do this, I simply rotate one of the rays in a given direction and check wether the angle gets bigger or smaller...
//	double3 rotAxis=cross(params.scatAxis,hitParams.normal); // calc axis around which these vectors are rotated to each other
//	//double3 eOut1Test=eOut1;
//	//rotateRay(eOut1Test, rotAxis, (double)0.01);
//	//double angleRay2Normal2=acos(dot(normalize(eOut1Test),hitParams.normal));
//	//if (angleRay2Normal2<angleRay2Normal)
//	//	angleRay2Normal=-angleRay2Normal;
//	// calc scattered deflection from specular reflection for first partial ray
//	// see http://web.physik.rwth-aachen.de/~roth/eteilchen/MonteCarlo.pdf slide 12 for reference. we need a better reference !!!
//	double fxi=0; // init 
//	double yi=1; // init
//	double xi; // declare
//	double BRDFmax=params.Kdl+params.Ksl+params.Ksp;
//	int count=0;
//	//params.sigmaXsp=0.209635407528088;
//	params.sigmaXsp=params.sigmaXsp/0.832554611157698; // renormalize from 50% sigma to 1/e sigma
//	for (count=0;count<20000;count++) //while (yi>=fxi) // this could turn into an endless loop !! maybe we should implement some break backup
//	{
//		// calc uniformly distributed number between -90 and 90
//		xi=Random(x1); // random between 0,1
//		xi=(xi-0.5)*PI-angleRay2Normal; // random between -pi/2-angleRay2Normal, pi/2-angleRay2Normal
//		// calc uniformly distributed number between 0 and max of scattering function
//		yi=Random(x1); // random between 0,1
//		yi=yi*BRDFmax; // random between 0 and maximum of scattering BRDF
//		// if xi is biger than pi/2 the probability of the diffuse lobe would become negative. This is not physical so we set its contribution to zero in this case...
//		if (abs(xi)>PI/2)
//		{
//			// calc probability of xi from scattering function
//			fxi=params.Ksl*expf(-pow((xi),2)/pow(params.sigmaXsl,2))+params.Ksp*expf(-pow((xi),2)/pow(params.sigmaXsp,2));
//		}
//		else
//		{
//			// calc probability of xi from scattering function
//			fxi=params.Kdl*cos(xi)+params.Ksl*expf(-pow((xi),2)/pow(params.sigmaXsl,2))+params.Ksp*expf(-pow((xi),2)/pow(params.sigmaXsp,2));
//		}
//		if (yi<=fxi)
//			break;
//	}
//
////	rotateRay(eOut1, rotAxis, xi);
//	rotateRay(eIn1, rotAxis, xi);
////	prd.direction=eOut1+eOut2;
//	prd.direction=eIn1+eIn2;
//	prd.flux=prd.flux*BRDFmax; // set flux to maximum of the BRDF. This way we don't have to throw away most of the rays. The further variation of the flux dependent on reflection angle is modelled via the number of rays generated in each direction...
//	return true;
//
//	//double cosPhi1=dot(normalize(eOut1),-hitParams.normal);
//	//double cosPhi2=cosPhi1*cos(xi)-sin(xi)*sqrt(1-cosPhi1*cosPhi1);
//	//double p=2*dot(eIn1,params.scatAxis);
//	//double q=length(eIn1)-pow(cosPhi1,2)/pow(cosPhi2,2);
//	//double Gamma=-p/2-sqrt(p*p/4-q);
//
//	//double testPhiNormal=acos(dot(hitParams.normal,make_double3(0,0,1)))/(2*PI)*360;
//	//double testPhi1=acos(cosPhi1)/(2*PI)*360;
//	//double testPhi2=acos(cosPhi2)/(2*PI)*360;
//	//double testAlpha=xi/(2*PI)*360;
//	//if (p*p<4*q)
//	//	double test = 0;
//
//
//	//if (xi<0)
//	//	eOut1=eOut1+Gamma*params.scatAxis;
//	//else
//	//	eOut1=eOut1-Gamma*params.scatAxis;
//	//eOut1=eOut1*length(eIn1)/length(eOut1);
//	//prd.direction=eOut1+eOut2;
//	//if (abs(length(prd.direction)-1)>0.0000000001)
//	//{
//	//	double test=length(prd.direction);
//	//	test=test+1;
//	//}
//	//return true;
//
//	/* see JOSA62 Vol.52 pp 672, Spencer, General Ray Tracing Procedure for reference of algorithm */
//	//double p=2*dot(prd.direction,hitParams.normal);
//	//double Delta=sin((double)50/360*2*PI);//sin(xi);//
//
//	//double testAlpha=xi/(2*PI)*360;
//	//double testSpecAngle=angleRay2Normal/(2*PI)*360;
//	////Delta=0;
//	//double q=(Delta*Delta-2*Delta*dot(ray.direction,params.scatAxis);
//	//double tau=-p/2-sqrt(p*p/4-q); // if we have reflection
//	//if (p*p<4*q)
//	//	double test = 0;
//	//// update ray direction
//	//prd.direction=prd.direction-Delta*params.scatAxis+tau*hitParams.normal;
//	//normalize(prd.direction);
//	//return true;
//
//};

#endif
