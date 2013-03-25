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

/**\file Pupil_RayAim_FarField_aim.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef PUPIL_RAYAIM_FARFIELD_AIM_H
#define PUPIL_RAYAIM_FARFIELD_AIM_H

#include "rayTracingMath.h"
#include "Pupil_aim.h"

/* declare class */
/**
  *\class   Pupil_RayAim_FarField_RedParams 
  *\ingroup Pupil
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
class Pupil_RayAim_FarField_RedParams : public Pupil_RedParams
{
public:

};

/**
 * \detail aimRayAimFarField 
 *
 * 
 *
 * \param[in] 
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool aimRayAimFarField(double3 position, double3 &direction, Pupil_RayAim_FarField_RedParams aimParams, unsigned long long iX, unsigned long long iY)
{
	return true;
}

#endif


