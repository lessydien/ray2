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

/**\file IntensityFieldStack.cpp
* \brief Intensity representation of light field
* 
*           
* \author Mauch
*/

#include "IntensityField_Stack.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"

/**
 * \detail addField 
 *
 * adds a field to the field stack
 *
 * \param[in] RayField* rayFieldPtr, double runningParam, long long index
 * 
 * \return fieldStackError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldStackError IntensityFieldStack::addField(IntensityField *fieldPtr, double runningParamIn, long long index)
{
	this->IntFieldList[index]=fieldPtr;
	return FIELDSTACK_NO_ERR;
};

/**
 * \detail getField 
 *
 * returns the field of given index from the field stack
 *
 * \param[in] long long index
 * 
 * \return IntensityField*
 * \sa 
 * \remarks 
 * \author Mauch
 */
IntensityField* IntensityFieldStack::getField(long long index)
{
	return this->IntFieldList[index];
};

/**
 * \detail write2TextFile
 *
 * saves the field to a textfile format
 *
 * \param[in] FILE* hfile
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError IntensityFieldStack::write2TextFile(FILE* hFile, detParams &oDetParams)
{
	return FIELD_NO_ERR;
};