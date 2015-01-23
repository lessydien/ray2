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

/**\file GeometricRenderField_hostDevice.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GEOMENDERFIELDHOSTDEVICE_H
  #define GEOMENDERFIELDHOSTDEVICE_H

#include "..\rayData.h"
#include <optix.h>
#include "macrosim_types.h"
#include <vector_types.h>


/**
* \detail initCPUSubset 
*
* \param[in] void
* 
* \return fieldError
* \sa 
* \remarks 
* \author Mauch
*/
inline RT_HOSTDEVICE geomRenderRayStruct createRay(unsigned long long jx, unsigned long long jy, unsigned long long jRay, renderFieldParams &oRenderFieldParams, double nImmersion, unsigned int seed)
{

    geomRenderRayStruct ray;

	// width of ray field in physical dimension
	double physWidth=oRenderFieldParams.rayPosEnd.x-oRenderFieldParams.rayPosStart.x;
	// height of ray field in physical dimension
	double physHeight=oRenderFieldParams.rayPosEnd.y-oRenderFieldParams.rayPosStart.y;
	// increment of rayposition in x and y in case of GridRect definition 
	double deltaW=0;
	double deltaH=0;
	// calc centre of ray field 
	double2 rayFieldCentre=make_double2(oRenderFieldParams.rayPosStart.x+physWidth/2,oRenderFieldParams.rayPosStart.y+physHeight/2);
	// declar variables for randomly distributing ray directions via an importance area
	double2 impAreaHalfWidth;
	double3 dirImpAreaCentre, tmpPos, impAreaRoot;
	double impAreaX, impAreaY, r, theta;
	double3 impAreaAxisX, impAreaAxisY;

    uint32_t x[5];
    RandomInit(seed, x);

    // create seed
    ray.currentSeed=(uint)BRandom(x);


	long long index=0; // loop counter for random rejection method

	ray.flux=1;
    ray.cumFlux=0;
    ray.secondary=false;
    ray.secondary_nr=0;
	ray.depth=0;
	ray.running=true;
	ray.currentGeometryID=0;
	ray.lambda=oRenderFieldParams.lambda;
	ray.nImmersed=nImmersion;
	ray.opl=0;

	// declare variables for placing a ray randomly inside an ellipse
	double ellipseX;
	double ellipseY;
	double3 exApt;
	double3 eyApt;

	// create rayposition in local coordinate system according to distribution type
	ray.position.z=0; // all rays start at z=0 in local coordinate system
	// calc increment along x- and y-direction
	if (oRenderFieldParams.width>0)
		deltaW= (physWidth)/double(oRenderFieldParams.width);
    else
    {
        cout << "error in GeometricRenderField.createRay: negative width is not allowed. \n";
        ray.running=false;
    }
	if (oRenderFieldParams.height>0)
		// multiple directions per point are listed in y-direction. Therefore the physical height of the rayfield is different from the height of the ray list. This has to be considered here...
		deltaH= (physHeight)/double(oRenderFieldParams.height);
    else
    {
        cout << "error in GeometricRenderField.createRay: negative width is not allowed. \n";
        ray.running=false;
    }
	ray.position.x=oRenderFieldParams.rayPosStart.x+deltaW/2+jx*deltaW;
	ray.position.y=oRenderFieldParams.rayPosStart.y+deltaH/2+jy*deltaH;

	if(oRenderFieldParams.width*oRenderFieldParams.height==1)
	{
		ray.position=oRenderFieldParams.rayPosStart;
	}
	// transform rayposition into global coordinate system
	ray.position=oRenderFieldParams.Mrot*ray.position+oRenderFieldParams.translation;

    // create ray direction 
	aimRayTowardsImpArea(ray.direction, ray.position, oRenderFieldParams.importanceAreaRoot, oRenderFieldParams.importanceAreaHalfWidth, oRenderFieldParams.importanceAreaTilt, oRenderFieldParams.importanceAreaApertureType, ray.currentSeed);

    // save current seed to ray
	ray.currentSeed=x[4];//(uint)BRandom(x);

	return ray;
};

#endif

