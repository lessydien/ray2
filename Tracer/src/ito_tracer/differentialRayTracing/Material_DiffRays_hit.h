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

/**\file Material_DiffRays_hit.h
* \brief header file containing the definition of Mat_DiffRays_hitParams
* 
*           
* \author Mauch
*/

#ifndef MATERIAL_DIFFRAYS_HIT_H
#define MATERIAL_DIFFRAYS_HIT_H

#include "../Material_hit.h"

/* declare class */
/**
  *\class   Mat_DiffRays_hitParams
  *\ingroup Material
  *\brief   set of params that the hit program needs from the underlying geometry for differential rays
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
class Mat_DiffRays_hitParams : public Mat_hitParams
{
public:
	double3 mainDirX;
	double3 mainDirY;
	double2 mainRad;
};

#endif


