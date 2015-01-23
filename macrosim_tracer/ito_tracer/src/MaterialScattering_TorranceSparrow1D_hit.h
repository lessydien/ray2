#ifndef MATERIALSCATTERING_TORRANCESPARROW1D_HIT_H
  #define MATERIALSCATTERING_TORRANCESPARROW1D_HIT_H
  
#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "rayData.h"
#include <time.h>                      // define time()

#ifndef PI
	#define PI 3.14159265358979323846
#endif

class MatTorranceSparrow1D_params
{
public:
	double Kdl; // coefficient of diffuse lobe
	double Ksl; // coefficient of specular lobe
	double Ksp; // coefficient of specular peak
	double sigmaXsl; // width parameter of specular lobe
	double sigmaXsp; // width parameter of specular peak
	double3 scatAxis;
};

inline RT_HOSTDEVICE bool hitTorranceSparrow1D(rayStruct &prd, double3 p_normal, MatTorranceSparrow1D_params params)
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
	double specAngle=acos(dot(normalize(eOut1),-p_normal)); // calc angle of specular reflection to normal
	// we're intereseted in reflection no matter wether we hit the front or back surface of the geometry. so the angle must be smaller than  pi/2
	//if (abs(specAngle)>PI/2)
	//	specAngle=acos(dot(normalize(eOut1),p_normal)); // calc angle of specular reflection to normal
	// here we need to determine the sign of the specular angle. Because I can't think of any smart way to do this, I simply rotate one of the rays in a given direction and check wether the angle gets bigger or smaller...
	double3 rotAxis=cross(params.scatAxis,p_normal); // calc axis around which these vectors are rotated to each other
	//double3 eOut1Test=eOut1;
	//rotateRay(eOut1Test, rotAxis, (double)0.01);
	//double specAngle2=acos(dot(normalize(eOut1Test),p_normal));
	//if (specAngle2<specAngle)
	//	specAngle=-specAngle;
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

	rotateRay(eOut1, rotAxis, xi);
	prd.direction=eOut1+eOut2;
	prd.flux=prd.flux*BRDFmax; // set flux to maximum of the BRDF. This way we don't have to throw away most of the rays. The further variation of the flux dependent on reflection angle is modelled via the number of rays generated in each direction...
	return true;

	//double cosPhi1=dot(normalize(eOut1),-p_normal);
	//double cosPhi2=cosPhi1*cos(xi)-sin(xi)*sqrt(1-cosPhi1*cosPhi1);
	//double p=2*dot(eIn1,params.scatAxis);
	//double q=length(eIn1)-pow(cosPhi1,2)/pow(cosPhi2,2);
	//double Gamma=-p/2-sqrt(p*p/4-q);

	//double testPhiNormal=acos(dot(p_normal,make_double3(0,0,1)))/(2*PI)*360;
	//double testPhi1=acos(cosPhi1)/(2*PI)*360;
	//double testPhi2=acos(cosPhi2)/(2*PI)*360;
	//double testAlpha=xi/(2*PI)*360;
	//if (p*p<4*q)
	//	double test = 0;


	//if (xi<0)
	//	eOut1=eOut1+Gamma*params.scatAxis;
	//else
	//	eOut1=eOut1-Gamma*params.scatAxis;
	//eOut1=eOut1*length(eIn1)/length(eOut1);
	//prd.direction=eOut1+eOut2;
	//if (abs(length(prd.direction)-1)>0.0000000001)
	//{
	//	double test=length(prd.direction);
	//	test=test+1;
	//}
	//return true;

	/* see JOSA62 Vol.52 pp 672, Spencer, General Ray Tracing Procedure for reference of algorithm */
	//double p=2*dot(prd.direction,p_normal);
	//double Delta=sin((double)50/360*2*PI);//sin(xi);//

	//double testAlpha=xi/(2*PI)*360;
	//double testSpecAngle=specAngle/(2*PI)*360;
	////Delta=0;
	//double q=(Delta*Delta-2*Delta*dot(ray.direction,params.scatAxis);
	//double tau=-p/2-sqrt(p*p/4-q); // if we have reflection
	//if (p*p<4*q)
	//	double test = 0;
	//// update ray direction
	//prd.direction=prd.direction-Delta*params.scatAxis+tau*p_normal;
	//normalize(prd.direction);
	//return true;

};

#endif
