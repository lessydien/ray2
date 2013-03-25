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

/**\file DetectorParams.h
* \brief header file that contains the detectorParams class
* 
*           
* \author Mauch
*/

/**
 *\defgroup Material
 */

#ifndef PUPILPARAMS_H
  #define PUPILPARAMS_H


typedef enum 
{
  PUP_ILLU,
  PUP_RAYAIM_FARFIELD,
  PUP_RAYAIM_IMAGE,
  PUP_UNKNOWN
} pupilType;

typedef struct
{
  double2 apertureHalfWidth;
  double3 root;
  long long pupilID;
  double3 tilt;
  ApertureType apertureType;
  pupilType type;
} PupilParseParamStruct;


#endif
