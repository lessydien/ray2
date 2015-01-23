
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix.h>
#include "optix_math_new.h"
#include <sutilCommonDefines.h>
//#include "helpers.h"
#include "rayData.h"
#include "rayTracingMath.h"
#include "MaterialScattering_TorranceSparrow1D_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(double3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(MatTorranceSparrow1D_params, params, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__device__ void torranceSparrow1D_anyHit_device()
{
  if (prd.currentGeometryID == geometryID)
  {
    rtIgnoreIntersection();
  }

}

__device__ void torranceSparrow1D_closestHit_device( double3 p_normal, double t_hit )
{

  prd.position = prd.position + (t_hit) * prd.direction;
  prd.currentGeometryID=geometryID;

  if (prd.depth < max_depth)
  {
 
	uint32_t x1[5];
	RandomInit(prd.currentSeed, x1); // seed random generator
	prd.currentSeed=x1[4]; // save seed for next round
	// project incoming ray onto plane containing geometry orientation and scattering axis
	double3 eInB=p_normal*dot(prd.direction,p_normal); // calc projection of incoming ray on normal
	double3 eInA=params.scatAxis*dot(prd.direction,params.scatAxis); // calc projection of incoming ray on scattering axis
	double3 eIn1=eInA+eInB; // part of direction ray in plane containing geometry normal and scattering axis
	double3 eIn2=prd.direction-eIn1; // part of direction ray in plane containing geometry normal and non scattering axis
	double3 eOut2=reflect(eIn2,p_normal); // second partial ray is reflected conventionally
	double3 eOut1=reflect(normalize(eIn1),p_normal); // init eOut1 with specular reflection
	double specAngle=acosf(dot(normalize(eOut1),-p_normal)); // calc angle of specular reflection to normal
	// calc scattered deflection from specular reflection for first partial ray
	// see http://web.physik.rwth-aachen.de/~roth/eteilchen/MonteCarlo.pdf slide 12 for reference. we need a better reference !!!
	double fxi=0; // init 
	double yi=1; // init
	double xi; // declare
	double BRDFmax=params.Kdl+params.Ksl+params.Ksp;
	int count=0;
	for (count=0;count<2000;count++) //while (yi>=fxi) // this could turn into an endless loop !! maybe we should implement some break backup
	{
		// calc uniformly distributed number between -90 and 90
		xi=Random(x1); // random between 0,1
		xi=(xi-0.5)*PI-specAngle; // random between -pi/2-specAngle, pi/2-specAngle
		// calc uniformly distributed number between 0 and max of scattering function
		yi=Random(x1); // random between 0,1
		yi=yi*BRDFmax; // random between 0 and maximum of scattering BRDF
		// if xi is biger than pi/2 the probability of the diffuse lobe would become negative. This is not physical so we set its contribution to zero in this case...
		if (abs(xi)>PI/2)
		{
			// calc probability of xi from scattering function
			fxi=params.Ksl*expf(-params.sigmaXsl*pow((xi),2))+params.Ksp*expf(-params.sigmaXsp*pow((xi),2));
		}
		else
		{
			// calc probability of xi from scattering function
			fxi=params.Kdl*cos(xi)+params.Ksl*expf(-params.sigmaXsl*pow((xi),2))+params.Ksp*expf(-params.sigmaXsp*pow((xi),2));
		}
		if (yi<=fxi)
			break;
	}
	double3 rotAxis=cross(params.scatAxis,p_normal); // calc axis around which these vectors are rotated to each other
	rotateRay(eOut1, rotAxis, xi);
	prd.direction=eOut1+eOut2;
	prd.flux=prd.flux*BRDFmax; // set flux to maximum of the BRDF. This way we don't have to throw away most of the rays. The further variation of the flux dependent on reflection angle is modelled via the number of rays generated in each direction...


    // the usual inline RT_HOSTDEVICE procedure is not possible here. The nvcc compiler doesn't seem to be able to handle loops inside inline functions...
    //hitTorranceSparrow1D(prd, p_normal, params);

	//uint32_t x1[5];
	//RandomInit(prd.currentSeed, x1); // seed random generator
	//prd.currentSeed=x1[4]; // save seed for next round
	///* chose diffraction order to reflect into */
	//// see http://web.physik.rwth-aachen.de/~roth/eteilchen/MonteCarlo.pdf slide 12 for reference. we need a better reference !!!
	//double BRDFmax=params.Kdl+params.Ksl+params.Ksp;
	//double fxi=0; // init 
	//double yi=1; // init
	//double xi; // declare
	//int count=0;
	//for (count=0;count<2000;count++) //while (yi>=fxi) // this could turn into an endless loop !! maybe we should implement some break backup
	//{
	//	// calc uniformly distributed number between -1 and 1
	//	xi=(Random(x1)-0.5)*PI; // random between -pi/2,pi/2
	//	// calc uniformly distributed number between 0 and max of scattering function
	//	yi=Random(x1); // random between 0,1
	//	yi=yi*BRDFmax; // random between 0 and maximum of scattering BRDF
	//	// if xi is biger than pi/2 the probability of the diffuse lobe would become negative. This is not physical so we set its contribution to zero in this case...
	//	if (abs(xi)>PI/2)
	//	{
	//		// calc probability of xi from scattering function
	//		fxi=params.Ksl*expf(-params.sigmaXsl*pow((xi),2))+params.Ksp*expf(-params.sigmaXsp*pow((xi),2));
	//	}
	//	else
	//	{
	//		// calc probability of xi from scattering function
	//		fxi=params.Kdl*cos(xi)+params.Ksl*expf(-params.sigmaXsl*pow((xi),2))+params.Ksp*expf(-params.sigmaXsp*pow((xi),2));
	//	}
	//	if (yi<=fxi)
	//		break;
	//}
	///* see JOSA62 Vol.52 pp 672, Spencer, General Ray Tracing Procedure for reference of algorithm */
	//double p=2*dot(prd.direction,p_normal)/dot(p_normal,p_normal);
	//double Delta=sin(xi);//sin((double)10/360*2*PI);
	////Delta=0;
	//double q=(Delta*Delta-2*Delta*(prd.direction.x+prd.direction.y*params.scatAxis.y+prd.direction.z*params.scatAxis.z))/dot(p_normal,p_normal);
	//double tau=-p/2-sqrt(p*p/4-q); // what do we do if we have an imaginary root ??
	//// update ray direction
	//prd.direction=prd.direction-Delta*params.scatAxis+tau*p_normal;
	//normalize(prd.direction);

  }
  else
  {
    prd.running=false; //stop ray
  }
  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  torranceSparrow1D_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  torranceSparrow1D_closestHit_device( geometric_normal, t_hit ); 
}
