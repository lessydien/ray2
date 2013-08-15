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

/**\file MaterialDOE_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Bielke
*/

#ifndef MATERIALDOE_HIT_H
#define MATERIALDOE_HIT_H

#include "rayTracingMath.h"
//#include "differentialRayTracing/Material_DiffRays_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

class MatDOE_coeffVec
{
public:
	double data[44];
};

class MatDOE_lookUp
{
public:
	double data[20*30*21];
};

/* declare class */
/**
  *\class   MatDOE_params 
  *\ingroup Material
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
class MatDOE_hitPparams : public Mat_hitParams
{
public:
	double n1; // refractive index1
	double n2; // refractive index2
	//double t; // amplitude transmission coefficient
	//double r; // amplitude reflection coefficient
};

/**
  *\class   MatDOE_params 
  *\ingroup Material
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
class MatDOE_params
{
public:
	double3 geomRoot;
	double3 geomTilt;
	double stepHeight;
	short coeffVecLength;
	short3 effLookUpTableDims; // dimensions of lookup table: x=z, y=s, z=i
	int dOEnr;
	double n1; 
	double n2; 
};

/**
 * \detail hitDOE 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, MatDOE_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitDOE(rayStruct &ray, Mat_hitParams hitParams, MatDOE_params params, MatDOE_coeffVec &coeffVec, MatDOE_lookUp &effLookUpTable, double t_hit, int geomID, bool coat_reflected)
//inline RT_HOSTDEVICE bool hitDOE(rayStruct &ray, Mat_hitParams hitParams, MatDOE_params params, double t_hit, int geomID, bool coat_reflected)
{
	ray.position=ray.position+t_hit*ray.direction;
	ray.currentGeometryID=geomID;
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	
	double3 root = params.geomRoot;
	double3 tilt = params.geomTilt;
	double3 tiltrad = params.geomTilt*2*PI/360;

	double IndexBefore = 1;
	double IndexAfter = 1;


	int nCoefficients = params.coeffVecLength -1;

	bool firstOrder = false;
	double normRad = coeffVec.data[0];
	if (normRad < 0)
	{
		firstOrder = true;
		normRad = -normRad;
	}

		/* Polynomial exponents as numbered in Zemax !!!Currently only 54 orders!!! */
	int expX[54] = {1,0,2,1,0,3,2,1,0,4,3,2,1,0,5,4,3,2,1,0,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0,8,7,6,5,4,3,2,1,0,9,8,7,6,5,4,3,2,1,0};
	int expY[54] = {0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,9};


	//Oberflächennormale für DOE
	double3 n = make_double3(0,0,1);

	// transform ray to local coordinate system
	double3 zeroposition=ray.position-params.geomRoot;
	rotateRayInv(&zeroposition,tiltrad);
	double3 zerodirection=ray.direction;
	rotateRayInv(&zerodirection,tiltrad);

	// mirror x-coordinate
	//zeroposition.y=-zeroposition.y;

	double3 Local;

	if (params.n1 == ray.nImmersed)
		{
			IndexBefore = params.n1;
			IndexAfter = params.n2;
		}
	else if (params.n2 == ray.nImmersed)
		{
			IndexBefore = params.n2;
			IndexAfter = params.n1;
		}
	else
		{
			return false;
		}
	
	bool refractbefore = false;

	if (IndexBefore > IndexAfter) refractbefore = true;
	
	if (ray.nImmersed==params.n1 && refractbefore == true)
	{
		if (calcSnellsLaw(&(zerodirection),make_double3(0,0,1), ray.nImmersed, params.n2))
			ray.nImmersed=params.n2;
	}
	else if (ray.nImmersed==params.n2 && refractbefore == true)
	{
		if (!calcSnellsLaw(&(zerodirection),make_double3(0,0,1), ray.nImmersed, params.n1))
			ray.nImmersed=params.n1;
	}


	Local.x = zeroposition.x; // local x and y coordinate in lens units
	Local.y = zeroposition.y;
	Local.z = 0;

	
	//double stepheight = (ray.lambda*1000000)/params.stepHeight;
	int orders[21] = {1,0,2,-1,-2,3,4,5,6,7,8,9,10,-3,-4,-5,-6,-7,-8,-9,-10};

	double prob_vector[21];
	double prob_vector_temp[21];
	
	//double lambda_real = 2*stepheight*(params.n2-params.n1);
	double lambda_real = 2*params.stepHeight*fabs(params.n2-params.n1)/1000000;
	if(lambda_real == 0)
	{ 
		lambda_real = ray.lambda;
	}
	double height = ray.lambda/lambda_real;  // design wavelength / real wavelength
	int i,j;
	double p_,h_,dp,dh,e1,e2,e3,e4;

	//----------
	//lokale Phase + Ableitung + Gitterperiode bestimmen
	//----------

	double Lambda = ray.lambda ; // Vacuum wavelength in mm

	double relative_index = IndexBefore / IndexAfter;
	
	// Ableitung 
	double x, y;
	double Phase, dPdx, dPdy, dPdx_temp, dPdy_temp;
	double ConversionRadiansToWaves = 1/(2 * PI);
	double ConversionWavesToLensUnits = Lambda ;
	double ConversionFactor = ConversionRadiansToWaves * ConversionWavesToLensUnits; // from radians to lens units

		/* x and y in polinomial coordinate system */
	x = Local.x / normRad;
	y = Local.y / normRad;
	
		/* Initialize total phase and derivatives [rad] */
	Phase = 0;
	dPdx_temp = 0;
	dPdy_temp = 0;
			/* Calculate phase and derivatives from polynomial [rad] */
	for(i = 0; i < nCoefficients; i++)
	{		
		Phase += coeffVec.data[i+1] * pow(x,  expX[i]     ) * pow(y,  expY[i]     );
		if (expX[i] != 0) dPdx_temp += expX[i] * coeffVec.data[i+1] * pow(x, (expX[i] - 1)) * pow(y,  expY[i]     );
		if (expY[i] != 0) dPdy_temp += expY[i] * coeffVec.data[i+1] * pow(x,  expX[i]     ) * pow(y, (expY[i] - 1));			
	}
	
		/* Phase [lens units] and derivatives [dimensionless] */
	Phase = Phase * ConversionFactor;
	dPdx = dPdx_temp * ConversionFactor; 
	dPdy = dPdy_temp * ConversionFactor;


			/* Effective grating periods (2D in local coordinate system)*/
	double3 g;
	g.x  = dPdx / Lambda / normRad;
	g.y  = dPdy / Lambda / normRad;
	g.z  = 0;
	
		/* Project the grating on the surface (3D) */
	double g_projection = dot(g, n);
	double3 g_effective = g - g_projection * n;
		
		/* Calc the effective line density */
	double line_density = length(g_effective);
	double period = 1/line_density;	// Gitterperiode in mm 

	//----------
	//lokale Phase + Ableitung + Gitterperiode bestimmen FERTIG
	//----------


	//----------
	//Effizienzenverteilung + Ordnung bestimmten
	//----------

	// effLookUpTable is organized as an 1D-array
	// efficiencies[z][s][i+10] in Prussens old code becomes effLookUpTable[s+z*(effLookUpTableDims.x+1)+(i+10)*effLookUpTableDims.x*effLookUpTableDims.y] here...

	if (abs(orders[0])>10)
	{
		return false; // as for now, don't consider orders greater than 10
	}
	
//	for(i=2;i<STUETZSTELLENH-1;i++)
	for(i=2;i<29;i++)
	{
		if (height<=effLookUpTable.data[0+i*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y]) break; // play it safe: always inside interpolation area...
	}
	
//	for(j=2;j<STUETZSTELLENP-1;j++) 
	for(j=2;j<19;j++) 
	{
		if (period<=effLookUpTable.data[j+0*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y]) break;
	}


	// Linear interpolation in lookup matrix
	e1=effLookUpTable.data[j+i*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
	e2=effLookUpTable.data[j+(i-1)*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
	e3=effLookUpTable.data[(j-1)+(i-1)*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
	e4=effLookUpTable.data[(j-1)+i*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];

	dh = effLookUpTable.data[0+i*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y] - effLookUpTable.data[0+(i-1)*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
	if (dh==0) 
	{
		return false; // warum auch immer das passieren sollte...
	}
	dp = effLookUpTable.data[j+0*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y] - effLookUpTable.data[(j-1)+0*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
	if (dp==0) 
	{
		return false; // warum auch immer das passieren sollte...
	}

	h_=(height-effLookUpTable.data[0+(i-1)*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y])/dh;
	
	p_=(period-effLookUpTable.data[(j-1)+0*params.effLookUpTableDims.x+(orders[0]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y])/dp;

					//   e2---e1
					//   |     |
					//   e3---e4

	prob_vector[0]= e3 + h_*(e4-e3) + p_*(e2-e3) + p_*h_*(e1-e2-e4+e3);

	int k;

	for (k=1;k<21;k++)
	{

			if (abs(orders[k])>10) return 0.0; // as for now, don't consider orders greater than 10
			for(i=2;i<29;i++)
			{
				if (height<=effLookUpTable.data[0+i*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y]) break; // play it safe: always inside interpolation area...
			}
			for(j=2;j<19;j++) 
			{
				if (period<=effLookUpTable.data[j+0*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y]) break;
			}

			e1=effLookUpTable.data[j+i*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
			e2=effLookUpTable.data[j+(i-1)*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
			e3=effLookUpTable.data[(j-1)+(i-1)*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
			e4=effLookUpTable.data[(j-1)+i*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
			dh = effLookUpTable.data[0+i*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y]-effLookUpTable.data[0+(i-1)*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
			if (dh==0) 
			{
				return false; // warum auch immer das passieren sollte...
			}
			dp = effLookUpTable.data[j+0*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y]-effLookUpTable.data[(j-1)+0*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y];
			if (dp==0) 
			{
				return false; // warum auch immer das passieren sollte...
			}
			h_=(height-effLookUpTable.data[0+(i-1)*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y])/dh;
			p_=(period-effLookUpTable.data[(j-1)+0*params.effLookUpTableDims.x+(orders[k]+10)*params.effLookUpTableDims.x*params.effLookUpTableDims.y])/dp;
			prob_vector_temp[k]= e3 + h_*(e4-e3) + p_*(e2-e3) + p_*h_*(e1-e2-e4+e3);
			prob_vector[k]=prob_vector[k-1] + prob_vector_temp[k];
	}

//std::cout << "order 1, efficiency: " << prob_vector[0] << std::endl;

uint32_t x1[5];
RandomInit(ray.currentSeed, x1); // init random variable
ray.currentSeed=x1[4]; // save seed for next round


double zufall = (Random(x1))*prob_vector[20];
//int Schrottordnung;


for (i=0;i<21;i++)
{
	if (zufall<=prob_vector[i]) break;
}

//if(zufall>prob_vector[20])
//	Schrottordnung=1;
//else
//	Schrottordnung=0;

double CurrentOrder = orders[i];
if (firstOrder == true)
{
	CurrentOrder = 1;
}

	//----------
	//Effizienzenverteilung + Ordnung bestimmten FERTIG
	//----------

	//----------
	//Richtungsvektor der festgelegten Ordnung bestimmen
	//----------

	//dPdy = dPdy / fabs(relative_index) * CurrentOrder / normRad;
	//dPdx = dPdx / fabs(relative_index) * CurrentOrder / normRad;

	dPdy = dPdy * CurrentOrder / normRad;
	dPdx = dPdx * CurrentOrder / normRad;

	//rotateRay(&zerodirection, params.geomTilt);
	



//	double nn = 1;

	// --->>>  rotateRayInv(&zerodirection, tiltrad);
	//rotateRay(&zerodirection, tiltrad);
	


	double3 u;
//	u.x = ray_in.x + nn * (dPdx);	//data[ 4]  = x cosine of specular ray, on output it is the scattered ray
	u.x = zerodirection.x + (dPdx);	//data[ 4]  = x cosine of specular ray, on output it is the scattered ray
//	u.y = ray_in.y + nn * (dPdy);	//data[ 5]  = y cosine of specular ray, on output it is the scattered ray
	u.y = zerodirection.y + (dPdy);	//data[ 5]  = y cosine of specular ray, on output it is the scattered ray
	u.z = zerodirection.z ;				//data[ 6]  = z cosine of specular ray, on output it is the scattered ray

	// --->>>  if (zerodirection.z < 0) n = -n;


	double rad = n.x*u.x + n.y*u.y + n.z*u.z;

	rad = 1.0 - (u.x*u.x + u.y*u.y + u.z*u.z) + rad*rad;
	if (rad <= 0.0) rad = 0.0;
		else rad = sqrt(rad);

	zerodirection.x= u.x - (n.x*u.x + n.y*u.y + n.z*u.z)*n.x + n.x * rad;
	zerodirection.y= u.y - (n.x*u.x + n.y*u.y + n.z*u.z)*n.y + n.y * rad;
	zerodirection.z= u.z - (n.x*u.x + n.y*u.y + n.z*u.z)*n.z + n.z * rad;

	double raydirectiontest = n.z*u.z;
	if (raydirectiontest < 0)
	{
		zerodirection.z = -zerodirection.z;
		
	}
	
	zerodirection = normalize(zerodirection);

	// --->>>  rotateRay(&ray.direction, tiltrad);
	// --->>>  rotateRay(&n, tiltrad);


	if (coat_reflected)
		ray.direction=reflect(zerodirection,make_double3(0,0,1));
	else
	{ 


		if (ray.nImmersed==params.n1 && refractbefore == false)
		{
			if (calcSnellsLaw(&(zerodirection),make_double3(0,0,1), ray.nImmersed, params.n2))
				ray.nImmersed=params.n2;
		}
		else if (ray.nImmersed==params.n2 && refractbefore == false)
		{
			if (calcSnellsLaw(&(zerodirection),make_double3(0,0,1), ray.nImmersed, params.n1))
				ray.nImmersed=params.n1;
		}
		//else
		//{
		//	// some error mechanism
		//	return false;
		//}
	}

	// transform back to 
	ray.direction=zerodirection;
	rotateRay(&ray.direction,tiltrad);

	return true;
}

#endif


