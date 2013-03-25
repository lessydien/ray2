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

/**\file Converter.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef CONVERTER_H
  #define CONVERTER_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "RayField.h"
#include "GeometricRayField.h"
#include "ScalarLightField.h"
#include "IntensityField.h"
#include "GaussBeamRayField.h"
#include "DiffRayField.h"
#include "converterMath.h"

typedef enum 
{
  CONVERTER_ERR,
  CONVERTER_NO_ERR
} converterError;

converterError convertGaussBeamRayField2ScalarLightFieldCPU(GaussBeamRayField *rayField, ScalarLightField *U);
converterError convertGeomRayField2IntensityCPU(GeometricRayField *geomRayFieldPtr, IntensityField *intensityFieldPtr);


#endif

