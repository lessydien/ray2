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

/**\file MicroLensArray.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MICROLENSARRAY_H
  #define MICROLENSARRAY_H
  
/* include header of basis class */
#include "Geometry.h"
#include "Material.h"
#include "pugixml.hpp"
#include "stdio.h"
#include <vector>
#include "my_vector_functions.h"

#include <optix.h>

using namespace std;



/* declare class */
/**
  *\class   MicroLensArray
  *\ingroup Geometry
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
class MicroLensArray : public virtual Geometry
{
  protected:

  private:

  public:
    /* standard constructor */
    MicroLensArray()
	{
	  materialList = NULL;
	  materialListLength = 0;
	}
    /* Constructor with known list length */
    MicroLensArray(int length)
	{
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	}
	/* Destruktor */
	~MicroLensArray()
	{
	  if (materialList!=NULL)
	  {
		  // delete all the materials attached to this geometry
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
				  materialList[i]=NULL;
			  }
		  }
		  delete materialList;
		  materialList = NULL;
	  }
	}

	geometryError parseXml(pugi::xml_node &geometry, simMode l_mode, vector<Geometry*> &geomVec);
};

#endif
