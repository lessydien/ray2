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

/**\file Converter.cpp
* \brief Wrapper of the functions for converting one field representation to another
* 
*           
* \author Mauch
*/

#include "Converter.h"
#include "myUtil.h"
#include <iostream>

converterError convertGaussBeamRayField2ScalarLightFieldCPU(GaussBeamRayField *rayFieldPtr, ScalarLightField *lightFieldPtr)
{
	if( !gaussBeams2ScalarFieldCPU(rayFieldPtr->getRayList(), rayFieldPtr->getRayListLength(), lightFieldPtr->getFieldPtr(), lightFieldPtr->getParamsPtr()) )
	{
		std::cout << "error in convertGaussBeamRayField2ScalarLightFieldCPU(): gaussBeams2ScalarFieldCPU() returned an error." << std::endl;
		return CONVERTER_ERR;
	}
	return CONVERTER_NO_ERR;
};

converterError convertGeomRayField2IntensityCPU(GeometricRayField *geomRayFieldPtr, IntensityField *intensityFieldPtr)
{
	if( !geomRays2IntensityCPU(geomRayFieldPtr->getRayList(), geomRayFieldPtr->getRayListLength(), intensityFieldPtr->getIntensityPtr(), intensityFieldPtr->getParamsPtr()->MTransform, intensityFieldPtr->getParamsPtr()->scale, intensityFieldPtr->getParamsPtr()->nrPixels, geomRayFieldPtr->getParamsPtr()->coherence) )
	{
		std::cout << "error in convertGeomRayField2IntensityCPU(): geomRays2IntensityCPU() returned an error." << std::endl;
		return CONVERTER_ERR;
	}
	return CONVERTER_NO_ERR;
};
