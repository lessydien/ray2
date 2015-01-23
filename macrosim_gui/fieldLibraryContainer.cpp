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

#include "fieldLibraryContainer.h"
#include "fieldItemLib.h"
#include <iostream>
using namespace std;

using namespace macrosim;

FieldLibContainer::FieldLibContainer(QObject *parent) :
	AbstractItem(FIELDCONTAINER, "Sources", parent)
{
	FieldItemLib l_fieldLib;
	QList<AbstractItem*> l_list=l_fieldLib.fillLibrary();
	for (int i=0; i<l_list.count(); i++)
	{
		m_childs.append(l_list.at(i));
	}
}

FieldLibContainer::~FieldLibContainer()
{
	m_childs.clear();
}