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

#include "macrosim_librarymodel.h"
#include "abstractItem.h"
#include "GeometryItem.h"
#include "MaterialItem.h"
#include "ScatterItem.h"
#include "CoatingItem.h"
#include "materialItemLib.h"
#include "coatingItemLib.h"
#include "scatterItemLib.h"
#include "miscItem.h"
#include "miscItemLib.h"

using namespace macrosim;
//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
LibraryModel::LibraryModel(QObject *parent=0) :
	SceneModel(parent)
{
}

LibraryModel::~LibraryModel()
{
}

void LibraryModel::appendItem(macrosim::AbstractItem* item, vtkSmartPointer<vtkRenderer> renderer, QModelIndex parentIndex, int rowIn) 
{
	GeometryItem* l_pGeom;
	MiscItem* l_pMiscItem;
	GeomGroupItem* l_pGeomGroupItem;
//	QModelIndex l_index;
	// get row to insert the item
	int row;
	if (parentIndex==QModelIndex())
		row=m_data.size();
	else
	{
		row=rowIn;
	}
	int coloumn=0;

	beginInsertRows(parentIndex, row, row);
	// only if we append at the top level, we need to actually append the item to the models data list
	if (parentIndex==QModelIndex())
		m_data.append(item);
	// create modelIndex at which the item was just inserted
	QModelIndex l_index=this->index(row, coloumn, parentIndex);
	// save this modelIndex to the appended item
	item->setModelIndex(l_index);
	endInsertRows();
	// now do the recursion over the items childs
	for (unsigned int i=0; i<item->getNumberOfChilds(); i++)
	{
		this->appendItem(item->getChild(i), renderer, l_index, i);
	}

}