#include "MaterialScattering_TorranceSparrow1D_hit.h"

RT_HOSTDEVICE bool hitTorranceSparrow1D(rayStruct &ray, double3 normal, MatTorranceSparrow1D_params params)
{
	uint32_t x1[5];
	RandomInit(ray.currentSeed, x1); // seed random generator
	ray.currentSeed=x1[4]; // save seed for next round
	// project incoming ray onto plane containing geometry orientation and scattering axis
	double3 eInB=normal*dot(ray.direction,normal); // calc projection of incoming ray on normal
	double3 eInA=params.scatAxis*dot(ray.direction,params.scatAxis); // calc projection of incoming ray on scattering axis
	double3 eIn1=eInA+eInB; // part of direction ray in plane containing geometry normal and scattering axis
	double3 eIn2=ray.direction-eIn1; // part of direction ray in plane containing geometry normal and non scattering axis
	double3 eOut2=reflect(eIn2,normal); // second partial ray is reflected conventionally
	double3 eOut1=reflect(eIn1,normal); // init eOut1 with specular reflection
	double specAngle=0;//acosf(dot(normalize(eOut1),-normal)); // calc angle of specular reflection to normal
	// we're intereseted in reflection no matter wether we hit the front or back surface of the geometry. so the angle must be smaller than  pi/2
	if (abs(specAngle)>PI/2)
		specAngle=0;//acosf(dot(normalize(eOut1),normal)); // calc angle of specular reflection to normal
	// here we need to determine the sign of the specular angle. Because I can't think of any smart way to do this, I simply rotate one of the rays in a given direction and check wether the angle gets bigger or smaller...
	double3 rotAxis=cross(params.scatAxis,normal); // calc axis around which these vectors are rotated to each other
	double3 eOut1Test=eOut1;
	rotateRay(eOut1Test, rotAxis, (double)0.1);
	double specAngle2=0;//asinf(dot(normalize(eOut1Test),-normal));
	if (specAngle2<specAngle)
		specAngle=-specAngle;
	// calc scattered deflection from specular reflection for first partial ray
	// see http://web.physik.rwth-aachen.de/~roth/eteilchen/MonteCarlo.pdf slide 12 for reference. we need a better reference !!!
	double fxi=0; // init 
	double yi=1; // init
	double xi; // declare
	double BRDFmax=params.Kdl+params.Ksl+params.Ksp;
	int count=0;
	for (count=0;count<20;count++) //while (yi>=fxi) // this could turn into an endless loop !! maybe we should implement some break backup
	{
		count++;
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
			fxi=params.Ksl*exp(-params.sigmaXsl*pow((xi),2))+params.Ksp*exp(-params.sigmaXsp*pow((xi),2));
		}
		else
		{
			// calc probability of xi from scattering function
			fxi=params.Kdl*cos(xi)+params.Ksl*exp(-params.sigmaXsl*pow((xi),2))+params.Ksp*exp(-params.sigmaXsp*pow((xi),2));
		}
		if (yi>=fxi)
			break;
	}
	rotateRay(eOut1, rotAxis, xi);
	ray.direction=eOut1+eOut2;
	ray.flux=ray.flux*BRDFmax; // set flux to maximum of the BRDF. This way we don't have to throw away most of the rays. The further variation of the flux dependent on reflection angle is modelled via the number of rays generated in each direction...
	return true;
};