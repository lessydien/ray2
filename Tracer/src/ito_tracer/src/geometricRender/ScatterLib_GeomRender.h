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

/**\file ScatterLib_GeomRender.h
* \brief header file that includes all the scatters defined in the application. If you define a new scatter, that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef SCATTERLIB_GEOMRENDER_H
  #define SCATTERLIB_GEOMRENDER_H

#include "../ScatterLib.h"
// include the individual materials
//#include "Scatter_TorranceSparrow1D_DiffRays.h"
//#include "Scatter_DoubleCauchy1D_DiffRays.h"
#include "Scatter_Lambert2D_GeomRender.h"
//#include "Scatter_Lambert2D_GeomRender.h"

class ScatterFab_GeomRender : public ScatterFab
{
protected:

public:

	ScatterFab_GeomRender()
	{

	}
	~ScatterFab_GeomRender()
	{

	}

	bool createScatInstFromXML(xml_node &node, Scatter* &pScat, SimParams simParams) const;
};

#endif
