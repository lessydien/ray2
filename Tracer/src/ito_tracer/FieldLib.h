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

/**\file FieldLib.h
* \brief header file that includes all the field representations defined in the application. If you define a new field representation that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef FIELDLIB_H
  #define FIELDLIB_H

// include the individual materials
#include "IntensityField.h"
#include "ScalarLightField.h"
#include "ScalarPlaneField.h"
#include "ScalarSphericalField.h"
#include "ScalarGaussianField.h"
#include "ScalarUserField.h"
#include "VectorLightField.h"
#include "GeometricRayField.h"
#include "GaussBeamRayField.h"
#include "DiffRayField.h"
#include "DiffRayField_RayAiming.h"
#include "DiffRayField_RayAiming_Holo.h"
#include "differentialRayTracing/DiffRayField_Freeform.h"
#include "PathTracingRayField.h"
#include "PhaseSpaceField.h"
#include "Parser_XML.h"

class FieldFab
{
protected:

public:

	FieldFab()
	{

	}
	~FieldFab()
	{

	}

	bool createFieldInstFromXML(xml_node &node, vector<Field*> &fieldVec) const;
};

#endif
