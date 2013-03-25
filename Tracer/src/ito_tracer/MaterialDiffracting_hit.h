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

/**\file MaterialDiffracting_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALDIFFRACTING_HIT_H
#define MATERIALDIFFRACTING_HIT_H

#include "rayTracingMath.h"
#include "Geometry_Intersect.h"

/* declare class */
/**
  *\class   MatDiffracting_params
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
class MatDiffracting_params
{
public:
	double n1; // refractive index1
	double n2; // refractive index2
	double2 importanceAreaHalfWidth;
	double3 importanceAreaRoot;
	double3 importanceAreaTilt;
	ApertureType importanceAreaApertureType;
	//double t; // amplitude transmission coefficient
	//double r; // amplitude reflection coefficient
};

/**
 * \detail hitDiffracting 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &prd, Mat_hitParams hitParams, MatDiffracting_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitDiffracting(rayStruct &prd, Mat_hitParams hitParams, MatDiffracting_params params, double t_hit, int geomID, bool coat_reflected)
{
	prd.position=prd.position+t_hit*prd.direction;
	prd.currentGeometryID=geomID;
	prd.opl=prd.opl+prd.nImmersed*t_hit;
	// check from which side the ray approaches
	if (prd.nImmersed==params.n1)
	{
		prd.nImmersed=params.n2;
	}
	else if (prd.nImmersed==params.n2)
	{
		prd.nImmersed=params.n1;
	}
	else
	{
		// some error mechanism
		return 0;
	}
	uint32_t x1[5];
	RandomInit(prd.currentSeed, x1); // init random variable

	// declare variables for randomly distributing points inside the importance area
	double impAreaX;
	double impAreaY;
	double3 exApt=make_double3(1,0,0);
	double3 eyApt=make_double3(0,1,0);
	double3 tmpPos;

	if (params.importanceAreaApertureType==AT_RECT)
	{
		impAreaX=(Random(x1)*2-1)*params.importanceAreaHalfWidth.x;
		impAreaY=(Random(x1)*2-1)*params.importanceAreaHalfWidth.y;
	}
	else // standard is elliptical
	{
		// place a point uniformingly randomly inside an elliptical importance area
		double theta=2*PI*Random(x1);
		double r=sqrt(Random(x1));
		impAreaX=params.importanceAreaHalfWidth.x*r*cos(theta);
		impAreaY=params.importanceAreaHalfWidth.y*r*sin(theta);
	}
	rotateRay(&exApt,params.importanceAreaTilt);
	rotateRay(&eyApt,params.importanceAreaTilt);
	tmpPos=params.importanceAreaRoot+impAreaX*exApt+impAreaY*eyApt;

	// calc ray direction as normalized vector connecting current ray position with point inside the importance area
	prd.direction=normalize(tmpPos-prd.position);
	prd.currentSeed=x1[4]; // save new seed for next randomization

	return true;
}

#endif


