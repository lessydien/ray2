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

#include "macrosim_scenemodel.h"
#include "AbstractItem.h"
#include "GeometryItem.h"
#include "MaterialItem.h"
#include "ScatterItem.h"
#include "CoatingItem.h"
#include "materialItemLib.h"
#include "coatingItemLib.h"
#include "scatterItemLib.h"

#include <QtOpenGL\qglfunctions.h>
#include "glut.h"

using namespace macrosim;
//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
SceneModel::SceneModel(QObject *parent)
	: QAbstractItemModel(parent),
	m_focusedItemIndex(QModelIndex()),
	m_geomID(0)
{
	m_headers << tr("Name") << tr("objectType");
	m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignRight);
}

SceneModel::~SceneModel()
{
   m_headers.clear();
   m_alignment.clear();
   m_data.clear();
   return;
}

int SceneModel::columnCount ( const QModelIndex &parent ) const
{
	return m_headers.size();
}

QVariant SceneModel::data ( const QModelIndex &index, int role ) const
{
	if (!index.isValid())
		return QVariant();
	else
	{
		AbstractItem* item = reinterpret_cast<AbstractItem*>(index.internalPointer());
		if (role == Qt::DisplayRole)
		{
			switch(index.column())
			{
			case 1:
				switch (item->getObjectType())
				{
				case AbstractItem::GEOMETRYGROUP:
					return tr("GeometryGroup");
					break;
				case AbstractItem::GEOMETRY:
					return tr("Geometry");
					break;
				case AbstractItem::MATERIAL:
					return tr("Material");
					break;
				case AbstractItem::COATING:
					return tr("Coating");
					break;
				case AbstractItem::SCATTER:
					return tr("Scatter");
					break;
				default:
					return QVariant();
					break;
				}
				break;
			case 0:
				return item->getName();
				break;
			default:
				return QVariant();
				break;
			}
		}
	}
	return QVariant();
}

QModelIndex SceneModel::index ( int row, int column, const QModelIndex &parent ) const
{
	if(parent.isValid()) //we are not in the root level
	{
		AbstractItem* parentItem = reinterpret_cast<AbstractItem*>(parent.internalPointer());
		if(parentItem != NULL)
		{
			if (row<0 || row>=parentItem->getNumberOfChilds())
				return QModelIndex();
			return createIndex(row, column, reinterpret_cast<void*>(parentItem->getChild(row)));
			//return createIndex(row, column, reinterpret_cast<void*>(&parentItem));
		}
		else
		{
			return QModelIndex();
		}
	}
	else //root level
	{
		if(row < 0 || row >= m_data.size())
		{
			return QModelIndex();
		}
		else
		{
			return createIndex(row, column, reinterpret_cast<void*>(m_data[row]));
		}
	}
}

QModelIndex	SceneModel::parent ( const QModelIndex & index ) const
{
	if (!index.isValid())
	{
		return QModelIndex();
	}

	//return QModelIndex();//index.parent();
	return getItemWithSpecificChild(NULL, reinterpret_cast<AbstractItem*>(index.internalPointer()));
	//return reinterpret_cast<AbstractItem*>(index.internalPointer())->getParent();
}

QModelIndex SceneModel::getItemWithSpecificChild(AbstractItem* rootItem, const AbstractItem* child) const
{
	QModelIndex temp;
	// if root item is not NULL, we are already in recursion
	if(rootItem != NULL)
	{
		for(unsigned int i=0;i<rootItem->getNumberOfChilds();i++)
		{
			// if one of the childs of the current root item equals the child we are looking for, we return the index to it as parent
			if(rootItem->getChild(i) == child)
			{
				return createIndex(i,0,reinterpret_cast<void*>(rootItem));
			}
			// 
			// if not, call recursion for the child with index i
			temp = getItemWithSpecificChild(rootItem->getChild(i), child);
			if(temp != QModelIndex())
			{
				return temp;
			}
		}
	}
	// the first time we call this function, the root item is NULL
	else
	{
		for(unsigned int i=0;i<m_data.size();i++)
		{
			if(m_data[i] == child)
			{
				// if one of the childs of our model equals the child we are looking for, we return an invalid index to signal that the child is in top level
				return QModelIndex();
			}
			
			// if not, call recursion for the child with index i
			temp = getItemWithSpecificChild(m_data[i], child);
			if(temp != QModelIndex())
			{
				return temp;
			}
		}
	}
	return QModelIndex();
}

int	SceneModel::rowCount ( const QModelIndex &parent ) const
{
	// if parent is invalid, we are at the root level here
	if (!parent.isValid())
	{
		return m_data.size();
	}
	else
	{
		return reinterpret_cast<AbstractItem*>(parent.internalPointer())->getNumberOfChilds();
	}

	return 0;
}


//----------------------------------------------------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
*
*/
QVariant SceneModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if( role == Qt::DisplayRole && orientation == Qt::Horizontal )
	{
		if(section >= 0 && section < m_headers.size())
		{
			return m_headers.at(section);
		}
		return QVariant();
	}
	return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
Qt::ItemFlags SceneModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

AbstractItem* SceneModel::getItem(const QModelIndex &index) const
{
	return (AbstractItem*) reinterpret_cast<AbstractItem*>(index.internalPointer());
}

void SceneModel::appendItem(macrosim::AbstractItem* item) 
{
	GeometryItem* l_pGeom;
	// assign geometryID if item is Geometry
	if (item->getObjectType() == AbstractItem::GEOMETRY)
	{
		m_geomID++;
		l_pGeom=reinterpret_cast<GeometryItem*>(item);
		l_pGeom->setGeometryID(m_geomID);
	}
	int row=m_data.size();
	// do the actual appending
	beginInsertRows(QModelIndex(), row, row);
	m_data.append(item);
	// create modelIndex at which the item was just inserted
	QModelIndex l_index=this->index(row,0,QModelIndex());
	// save this modelIndex to the appended item
	item->setModelIndex(l_index);
	// create modelIndices for childs of this item
	for (int i=0; i<item->getNumberOfChilds(); i++)
	{
		AbstractItem* l_pChild=item->getChild(i);
		QModelIndex l_childIndex=this->index(i,1,l_index);
		l_pChild->setModelIndex(l_childIndex);
		// create modelIndices for childs of this child
		for (int j=0; j<l_pChild->getNumberOfChilds(); j++)
		{
			AbstractItem* l_pChildChild=l_pChild->getChild(j);
			QModelIndex l_childChildIndex=this->index(j,2,l_childIndex);
			l_pChildChild->setModelIndex(l_childChildIndex);
		}
	}
	// emit rowsInserted-event ??
	endInsertRows();
	item->setFocus(false);
	// connect signals
	bool test=connect(item, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItemData(const QModelIndex &,  const QModelIndex &)));
	int bla=0;
}

void SceneModel::removeItem(QModelIndex index)
{
	if (index.isValid())
	{
		// get root item
		QModelIndex rootIndex=this->getRootIndex(index);
		// check wether root item is a geometry
		AbstractItem* l_pAbstractToBeRemoved=reinterpret_cast<AbstractItem*>(rootIndex.internalPointer());
		if (l_pAbstractToBeRemoved->getObjectType() == AbstractItem::GEOMETRY)
		{
			GeometryItem* l_pGeomToBeRemoved=reinterpret_cast<GeometryItem*>(l_pAbstractToBeRemoved);
			// adjust geomIDs of all geometries behind the geometry that that is about to be removed
			for (int i=index.row()+1;i<m_data.size();i++)
			{
				AbstractItem* l_pAbstract=reinterpret_cast<AbstractItem*>(m_data.at(i));
				if (l_pAbstract->getObjectType()==AbstractItem::GEOMETRY)
				{
					GeometryItem* l_pGeom=reinterpret_cast<GeometryItem*>(m_data.at(i));
					l_pGeom->setGeometryID(l_pGeom->getGeometryID()-1);
				}
			}
			// decrement internal geometry counter
			m_geomID--;
		}

		this->removeRow(index.row(),QModelIndex());
	}
}

void SceneModel::removeFocusedItem()
{
	this->removeItem(m_focusedItemIndex);
	// set item focus to none
	this->changeItemFocus(QModelIndex());
}

bool SceneModel::removeRows(int row, int count, const QModelIndex & parentInd)
{
	beginRemoveRows(parentInd, row, row+count-1);
	for (int i=row;i<row+count;i++)
	{
		if (i<m_data.size())
			m_data.removeAt(i);
	}
	endRemoveRows();
	return true;
}

// returns the modelIndex of the rootItem that ultimately holds the item with the given index
QModelIndex SceneModel::getRootIndex(const QModelIndex &index)
{
	QModelIndex rootIndex;
	rootIndex=parent(index);
	// if parent of index is invalid, we have found the root item and return it
	if (!rootIndex.isValid())
		return index;
	else
		return getRootIndex(rootIndex);
}

void SceneModel::changeItemData(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
	MaterialItemLib l_matLib;
	ScatterItemLib l_scatLib;
	CoatingItemLib l_coatLib;
	MaterialItem* l_pMatItem;
	ScatterItem* l_pScatItem;
	CoatingItem* l_pCoatItem;
	// check wether child of item changed
	AbstractItem* l_pAbstractItem=reinterpret_cast<AbstractItem*>(topLeft.internalPointer());
	switch (l_pAbstractItem->getObjectType())
	{
	case AbstractItem::GEOMETRY:
		l_pMatItem=reinterpret_cast<MaterialItem*>(l_pAbstractItem->getChild(0));
		// if the material changed, its modelIndex is invalid
		if (QModelIndex() == l_pMatItem->getModelIndex() )
		{
			// set model index of new material
			QModelIndex l_matIndex=this->index(0,1,topLeft);
			l_pMatItem->setModelIndex(l_matIndex);
		}
		break;
	case AbstractItem::MATERIAL:
		l_pMatItem=reinterpret_cast<MaterialItem*>(l_pAbstractItem);
		l_pScatItem=reinterpret_cast<ScatterItem*>(l_pMatItem->getScatter());
		l_pCoatItem=reinterpret_cast<CoatingItem*>(l_pMatItem->getCoating());
		// if coating changed, its modelIndex is invalid
		if (QModelIndex() == l_pCoatItem->getModelIndex() )
		{
			// set model index of new coating
			QModelIndex l_coatIndex=this->index(0,2,l_pMatItem->getModelIndex());
			l_pCoatItem->setModelIndex(l_coatIndex);
		}
		// if scatter changed, its modelIndex is invalid
		if (QModelIndex() == l_pScatItem->getModelIndex() )
		{
			// set model index of new scatter
			QModelIndex l_scatIndex=this->index(1,2,l_pMatItem->getModelIndex());
			l_pScatItem->setModelIndex(l_scatIndex);
		}
		break;

	default:
		break;
	}
	emit dataChanged(topLeft, bottomRight);
}


void SceneModel::changeItemFocus(const QModelIndex &index)
{
	// find rootItem that ultimately holds the item that just changed
	QModelIndex rootItemIndex=this->getRootIndex(index);

	if (rootItemIndex != m_focusedItemIndex)
	{
		if (rootItemIndex!=QModelIndex())
		{
			// if any item had focus before, remove focus from it now
			if (m_focusedItemIndex != QModelIndex())
				reinterpret_cast<AbstractItem*>(m_focusedItemIndex.internalPointer())->setFocus(false);

			// set new focus
			// all items in our model are AbstractItems, so we can savely cast here
			AbstractItem* l_pAbstractItem=this->getItem(rootItemIndex);
			l_pAbstractItem->setFocus(true);
		}
		m_focusedItemIndex=rootItemIndex;
		emit itemFocusChanged(index);
	}
}

void SceneModel::render(QMatrix4x4  &matrix, RenderOptions &options)
{
	for (int i=0; i < this->m_data.size(); i++)
		m_data.at(i)->render(matrix, options);
}