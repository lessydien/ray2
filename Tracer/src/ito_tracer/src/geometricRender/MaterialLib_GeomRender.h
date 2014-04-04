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

#ifndef MATERIALLIB_GEOMRENDER_H
  #define MATERIALLIB_GEOMRENDER_H

#include "../MaterialLib.h"

// include the individual materials
#include "MaterialLight_GeomRender.h"

class MaterialFab_GeomRender : public MaterialFab
{
protected:

public:

	MaterialFab_GeomRender()
	{

	}
	~MaterialFab_GeomRender()
	{

	}

	bool createMatInstFromXML(xml_node &node, Material* &pMat, SimParams simParams) const;
};

#endif
