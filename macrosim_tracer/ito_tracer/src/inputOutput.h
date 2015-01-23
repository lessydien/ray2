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

/**\file inputOutput.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

#include <stdio.h>
#include "rayData.h"
//#include "FieldLib.h"
#include "ScalarLightField.h"
#include "VectorLightField.h"
#include "IntensityField.h"
#include "PhaseSpaceField.h"


typedef enum
{
	IO_NO_ERR,
	IO_MATLAB_ERR,
	IO_FILE_ERR
}InputOutputError;

class rayDataOutParams
{
public:
	long long ID;
	bool reducedData;
};

InputOutputError writeGeomRayData2File(FILE* hFile, rayStruct* prdPtr, long long int length, rayDataOutParams params);
InputOutputError writeScalarField2File(FILE* hFile, ScalarLightField *ptrLightField);
InputOutputError writeIntensityField2File(FILE* hFile, IntensityField *ptrIntensityField);
InputOutputError writePhaseSpaceField2File(FILE* hFile, PhaseSpaceField *ptrPhaseSpaceField);

#endif