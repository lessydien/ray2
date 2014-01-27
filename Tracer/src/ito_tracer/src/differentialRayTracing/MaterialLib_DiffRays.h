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

/**\file MaterialLib_DiffRays.h
* \brief header file that includes all the materials defined in the application. If you define a new material, that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef MATERIALLIB_DIFFRAYS_H
  #define MATERIALLIB_DIFFRAYS_H

#include "../MaterialLib.h"

// include the individual materials
#include "MaterialReflecting_DiffRays.h"
#include "MaterialAbsorbing_DiffRays.h"
#include "MaterialRefracting_DiffRays.h"
#include "MaterialLinearGrating1D_DiffRays.h"
//#include "MaterialIdealLense_DiffRays.h"
#include "MaterialDiffracting_DiffRays.h"

class MaterialFab_DiffRays : public MaterialFab
{
protected:

public:

	MaterialFab_DiffRays()
	{

	}
	~MaterialFab_DiffRays()
	{

	}

	bool createMatInstFromXML(xml_node &node, Material* &pMat, SimParams simParams) const;
};

#endif
