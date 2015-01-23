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

/**\file CoatingLib.h
* \brief header file that includes all the coatings defined in the application. If you define a new coating that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef COATINGLIB_H
  #define COATINGLIB_H

// include the individual coatings
#include "Coating_NumCoeffs.h"
#include "Coating_DispersiveNumCoeffs.h"
#include "Coating_FresnelCoeffs.h"
#include "Coating_NoCoating.h"
#include "Parser_XML.h"

class CoatingFab
{
protected:

public:

	CoatingFab()
	{

	}
	~CoatingFab()
	{

	}

	virtual bool createCoatInstFromXML(xml_node &node, Coating* &pCoat, SimParams simParams) const;
};

#endif
