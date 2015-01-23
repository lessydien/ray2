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

#ifndef FIELDLIBCONTAINER
#define FIELDLIBCONTAINER

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "fieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldLibContainer
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class FieldLibContainer :
	public AbstractItem
{
	Q_OBJECT

public:

	FieldLibContainer(QObject *parent=0);
	~FieldLibContainer(void);

	//AbstractItem* getChild() const {return m_childs[0];};
	FieldItem* getChild(int i) const 
	{ 
		if (i>=m_childs.count())
			return NULL;
		else
		{
			if (m_childs[i]->getObjectType() != FIELD)
				return NULL;
			else
				return reinterpret_cast<FieldItem*>(m_childs[i]);
		}
	};

	virtual void appendChild(AbstractItem* child) 
	{
		if (child->getObjectType() == FIELD)
		{
			m_childs.append(child);
		}
	}


private:

};

}; //namespace macrosim

#endif