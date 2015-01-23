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

/**\file MaterialLib.h
* \brief header file that includes all the materials defined in the application. If you define a new material, that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef MATERIAL_DIFFRAYS_LIB_H
  #define MATERIAL_DIFFRAYS_LIB_H

// include the individual materials
#include "MaterialReflecting_DiffRays.h"
#include "MaterialAbsorbing_DiffRays.h"
#include "MaterialRefracting_DiffRays.h"
#include "MaterialLinearGrating1D_DiffRays.h"
#include "MaterialIdealLense_DiffRays.h"
#include "MaterialDiffracting_DiffRays.h"

//typedef enum 
//{
//  MT_MIRROR,
//  MT_NBK7,
//  MT_BK7,
//  MT_AIR,
//  MT_ABSORB,
////  MT_TORSPARR1D,
//  MT_LINGRAT1D,
//  MT_REFRMATERIAL,
//  MT_IDEALLENSE,
//  MT_DIFFRACT,
//  MT_UNKNOWNMATERIAL
//} MaterialType;

#endif
