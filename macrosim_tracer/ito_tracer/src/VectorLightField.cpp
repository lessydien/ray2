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

/**\file VectorLightField.cpp
* \brief vector field representation of the light field
* 
*           
* \author Mauch
*/

#include "VectorLightField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"

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
fieldError VectorLightField::write2TextFile(FILE* hFile, detParams &oDetParams)
{
	std::cout << "error in VectorLightField.write2TextFile(): not implemented yet" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail write2MatFile
 *
 * saves the field to a mat file
 *
 * \param[in] char* filename
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError VectorLightField::write2MatFile(char* filename, detParams &oDetParams)
{
	std::cout << "error in Field.write2MatFile(): not defined for the given field representation" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &field, vector<Field*> &fieldVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  VectorLightField::parseXml(pugi::xml_node &det, vector<Field*> &fieldVec, SimParams simParams)
{
	// call base class function
	if (FIELD_NO_ERR != Field::parseXml(det, fieldVec, simParams))
	{
		std::cout << "error in VectorLightField.parseXml(): Field.parseXml()  returned an error." << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};