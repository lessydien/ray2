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

/**\file Scatter_NoScatter.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_NOSCATTER_H
#define SCATTER_NOSCATTER_H

#include "Scatter.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"

/* declare class */
/**
  *\class   Scatter_NoScatter
  *\ingroup Scatter
  *\brief   
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
class Scatter_NoScatter: public Scatter
{
	protected:

  public:
    /* standard constructor */
    Scatter_NoScatter() :
		Scatter()
	{
	}

	ScatterError parseXml(pugi::xml_node &geometry, SimParams simParams);
};

#endif


