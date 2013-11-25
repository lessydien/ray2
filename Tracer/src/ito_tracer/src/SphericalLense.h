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

/**\file SphericalLense.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup SphericalLense
 */

#ifndef SPHERICALLENSE_H
  #define SPHERICALLENSE_H

#include "Geometry.h"
#include <optix.h>
#include "Material.h"
#include "DetectorParams.h"
#include "macrosim_functions.h"
#include "rayData.h"
#include "stdio.h"
#include "sampleConfig.h"
#include "pugixml.hpp"
#include <vector>

using namespace std;


/* declare class */
/**
  *\class   SphericalLense
  *\ingroup SphericalLense
  *\brief   base class of all geometries
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
class SphericalLense : public Geometry
{
  protected:

  private:
	

  public:
    /* standard constructor */
    SphericalLense()
	{
	  materialList=NULL;
	  materialListLength=0;
	}
    /* Constructor in case, list length is known */
    SphericalLense(int length)
	{
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	}
	/* Destruktor */
	virtual ~SphericalLense()
	{
	  if ( materialList != NULL)
	  {
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
			  }
		  }
		  delete materialList;
		  materialList = NULL;
	  }
	}

	geometryError parseXml(pugi::xml_node &geometry, TraceMode l_mode, vector<Geometry*> &geomVec);
};


#endif

