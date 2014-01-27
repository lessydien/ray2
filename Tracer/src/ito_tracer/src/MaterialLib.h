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

#ifndef MATERIALLIB_H
  #define MATERIALLIB_H

// include the individual materials
#include "MaterialReflecting.h"
#include "MaterialDOE.h"
#include "MaterialAbsorbing.h"
#include "MaterialRefracting.h"
#include "MaterialLinearGrating1D.h"
#include "MaterialIdealLense.h"
#include "MaterialDiffracting.h"
#include "MaterialReflecting_CovGlass.h"
#include "MaterialFilter.h"
#include "MaterialPathTraceSource.h"
#include "MaterialVolumeScatter.h"
#include "MaterialVolumeScatterBox.h"
#include "MaterialVolumeAbsorbing.h"
#include "Parser_XML.h"

class MaterialFab
{
protected:

public:

	MaterialFab()
	{

	}
	~MaterialFab()
	{

	}

	virtual bool createMatInstFromXML(xml_node &node, Material* &pMat, SimParams simParams) const;
};

#endif
