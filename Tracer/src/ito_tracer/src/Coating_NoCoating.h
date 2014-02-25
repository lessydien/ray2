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

/**\file Coating.h
* \brief 
* 
*           
* \author Mauch
*/


/**
 *\defgroup Coating
 */

#ifndef COATING_NOCOATING_H
  #define COATING_NOCOATING_H

#include <optix.h>
#include <optix_math.h>
#include <optix_host.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Coating_hit.h"
#include "stdlib.h"
#include "MaterialParams.h"
#include "pugixml.hpp"
#include <stdio.h>
#include "Coating.h"
//#include "CoatingLib.h"

/* declare class */
/**
  *\class   Coating_NoCoating 
  *\ingroup Coating
  *\brief   class of NoCoating
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class Coating_NoCoating : public Coating
{
  protected:
			
  public:
    bool update;

    Coating_NoCoating() :
		Coating()
	{
	}
	virtual ~Coating_NoCoating()
	{
		if (fullParamsPtr!=NULL)
		{
			delete fullParamsPtr;
			fullParamsPtr=NULL;
		}
	}
	virtual CoatingError parseXml(pugi::xml_node &node, SimParams simParams);
};

#endif

