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
#include "miscItem.h"
#include "miscItemLib.h"

//#include <QtOpenGL\qglfunctions.h>
////#include "glut.h"

using namespace macrosim;
//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
SceneModel::SceneModel(QObject *parent)
	: QAbstractItemModel(parent),
	m_focusedItemIndex(QModelIndex()),
	m_geomID(0),
	m_geomGroupNr(0),
	m_currentGeomGroupIndex(QModelIndex())
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
				case AbstractItem::MISCITEM:
					return tr("MiscItem");
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
				return rootItem->getModelIndex();//createIndex(i,0,reinterpret_cast<void*>(rootItem));
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

void SceneModel::exchangeItem(const QModelIndex &parentIndex, const int row, const int coloumn, macrosim::AbstractItem &pItem) 
{
	AbstractItem* l_pParentItem=reinterpret_cast<AbstractItem*>(parentIndex.internalPointer());

	this->beginInsertRows(parentIndex, row, row);
	QModelIndex l_index=this->index(row, 0, parentIndex);
	pItem.setModelIndex(l_index);
	this->endInsertRows();

	// recurse through childs
	for (int i=0; i<pItem.getNumberOfChilds(); i++)
	{
		this->exchangeItem(l_index, i, 0, *(pItem.getChild(i)));
	}
};

void SceneModel::appendItem(macrosim::AbstractItem* item, vtkSmartPointer<vtkRenderer> renderer, QModelIndex parentIndex, int rowIn) 
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
	bool geomToGeomGroup=false;
	bool geomGroupChanged=false;
	switch (item->getObjectType())
	{
	case AbstractItem::MISCITEM:
		l_pMiscItem=reinterpret_cast<MiscItem*>(item);
		switch (l_pMiscItem->getMiscType())
		{
		case MiscItem::GEOMETRYGROUP:
			l_pGeomGroupItem=reinterpret_cast<GeomGroupItem*>(l_pMiscItem);
			m_geomGroupNr++;
			geomGroupChanged=true;
			break;
		default:
			break;
		}
		break;
	case AbstractItem::GEOMETRY:
		l_pGeom=reinterpret_cast<GeometryItem*>(item);
		// if we don't have any geometrGroup yet, we create one
		if (m_geomGroupNr==0)
		{
			m_geomGroupNr++;
			// create the geometryGroup
			l_pGeomGroupItem=new GeomGroupItem();
			// now add the geometry to this group
			l_pGeomGroupItem->setChild(l_pGeom);
			// instead of appending the geometry to the model, we append the newly created geometryGroup
			// the geometry will be inserted inside the recursion of appending of the geometryGroup
			item=l_pGeomGroupItem;
			// signal creation of geometryGroup, so currentGeomGroupIndex will be set, once an index for the geometryGroup is created
			geomGroupChanged=true;
		}
		else
		{
			m_geomID++;
			l_pGeom->setGeometryID(m_geomID);
			// if this geometry was added from the library, it has no connection to a geometryGroup yet. therefore we create one here
			if (parentIndex==QModelIndex())
			{
				l_pGeomGroupItem=reinterpret_cast<GeomGroupItem*>(m_currentGeomGroupIndex.internalPointer());
				l_pGeomGroupItem->setChild(l_pGeom); // add the geometry
				row=l_pGeomGroupItem->getNumberOfChilds()-1; // index of geometry has to be created with the correct row
				// signal that a geometry has to be appended to the current geometryGroup
				geomToGeomGroup=true;
			}
		}
			
		break;
	default:
		break;
	}
	if (geomToGeomGroup)
		parentIndex=this->m_currentGeomGroupIndex;
	beginInsertRows(parentIndex, row, row);
	// only if we append at the top level, we need to actually append the item to the models data list
	if (parentIndex==QModelIndex())
		m_data.append(item);
	// create modelIndex at which the item was just inserted
	QModelIndex l_index=this->index(row, coloumn, parentIndex);
	// save this modelIndex to the appended item
	item->setModelIndex(l_index);
	endInsertRows();
	if (geomGroupChanged)
		this->m_currentGeomGroupIndex=l_index;
	// now do the recursion over the items childs
	for (unsigned int i=0; i<item->getNumberOfChilds(); i++)
	{
		this->appendItem(item->getChild(i), renderer, l_index, i);
	}

	// connect signals
	bool test=connect(item, SIGNAL(itemExchanged(const QModelIndex &, const int, const int, macrosim::AbstractItem &)), this, SLOT(exchangeItem(const QModelIndex &, const int, const int, macrosim::AbstractItem &)));	
	test=connect(item, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItemData(const QModelIndex &,  const QModelIndex &)));
	item->setRenderOptions(this->m_renderOptions);
	if (renderer)
		item->renderVtk(renderer);

	//GeometryItem* l_pGeom;
	//MiscItem* l_pMiscItem;
	//GeomGroupItem* l_pGeomGroupItem;
	//QModelIndex l_index;
	//int row=m_data.size();
	//int coloumn=0;
	//switch (item->getObjectType())
	//{
	//case AbstractItem::MISCITEM:
	//	l_pMiscItem=reinterpret_cast<MiscItem*>(item);
	//	switch (l_pMiscItem->getMiscType())
	//	{
	//	case MiscItem::GEOMETRYGROUP:
	//		l_pGeomGroupItem=reinterpret_cast<GeomGroupItem*>(l_pMiscItem);
	//		m_geomGroupNr++;
	//		// append geomGroup
	//		beginInsertRows(QModelIndex(), row, row);
	//		m_data.append(l_pGeomGroupItem);
	//		// create modelIndex at which the item was just inserted
	//		l_index=this->index(row, coloumn, QModelIndex());
	//		l_pGeomGroupItem->setModelIndex(l_index);
	//		// emit rowsInserted-event ??
	//		endInsertRows();
	//		for (unsigned int i=0; i<l_pGeomGroupItem->getNumberOfChilds(); i++)
	//		{
	//			beginInsertRows(l_pGeomGroupItem->getModelIndex(), i, i);
	//			l_pGeomGroupItem->getChild(i)->createModelIndex(l_pGeomGroupItem->getModelIndex(), i, 0);
	//			endInsertRows();
	//		}
	//		m_currentGeomGroupIndex=l_pGeomGroupItem->getModelIndex();
	//		break;
	//	default:
	//		break;
	//	}
	//	break;
	//case AbstractItem::GEOMETRY:
	//	m_geomID++;
	//	l_pGeom=reinterpret_cast<GeometryItem*>(item);
	//	l_pGeom->setGeometryID(m_geomID);
	//	// if we don't have any geometrGroup yet, we create one
	//	if (m_geomGroupNr==0)
	//	{
	//		m_geomGroupNr++;
	//		beginInsertRows(QModelIndex(), row, row);
	//		l_pGeomGroupItem=new GeomGroupItem();
	//		m_data.append(l_pGeomGroupItem);
	//		endInsertRows();
	//		l_index=this->index(row, coloumn, QModelIndex());
	//		l_pGeomGroupItem->setModelIndex(l_index);
	//		beginInsertRows(l_pGeomGroupItem->getModelIndex(), l_pGeomGroupItem->getNumberOfChilds(), l_pGeomGroupItem->getNumberOfChilds());
	//		l_pGeomGroupItem->setChild(l_pGeom);
	//		l_pGeom->createModelIndex(l_pGeomGroupItem->getModelIndex(), l_pGeomGroupItem->getNumberOfChilds()-1, 0);
	//		endInsertRows();
	//		m_currentGeomGroupIndex=l_pGeomGroupItem->getModelIndex();
	//		
	//	}
	//	else
	//	{
	//		// if not, we grab the current geometryGroup
	//		l_pGeomGroupItem=reinterpret_cast<GeomGroupItem*>(this->m_currentGeomGroupIndex.internalPointer());
	//		beginInsertRows(l_pGeomGroupItem->getModelIndex(), l_pGeomGroupItem->getNumberOfChilds(), l_pGeomGroupItem->getNumberOfChilds());
	//		l_pGeomGroupItem->setChild(l_pGeom);
	//		l_pGeom->createModelIndex(l_pGeomGroupItem->getModelIndex(), l_pGeomGroupItem->getNumberOfChilds()-1, coloumn); 
	//		endInsertRows();
	//	}		
	//		
	//	break;
	//default:
	//	beginInsertRows(QModelIndex(), row, row);
	//	m_data.append(item);
	//	// create modelIndex at which the item was just inserted
	//	QModelIndex l_index=this->createIndex(row, 0, item);
	//	// save this modelIndex to the appended item
	//	item->setModelIndex(l_index);
	//	endInsertRows();
	//	break;
	//}
	//// connect signals
	//connect(item, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItemData(const QModelIndex &,  const QModelIndex &)));

	//item->setRenderOptions(this->m_renderOptions);
	//if (renderer)
	//	item->renderVtk(renderer);

// ************************ very old ************************************************************
	//GeometryItem* l_pGeom;
	//// assign geometryID if item is Geometry
	//if (item->getObjectType() == AbstractItem::GEOMETRY)
	//{
	//	m_geomID++;
	//	l_pGeom=reinterpret_cast<GeometryItem*>(item);
	//	l_pGeom->setGeometryID(m_geomID);
	//}
	//int row=m_data.size();
	//// do the actual appending
	//beginInsertRows(QModelIndex(), row, row);
	//m_data.append(item);
	//// create modelIndex at which the item was just inserted
	//QModelIndex l_index=this->createIndex(row, 0, item);
	//// save this modelIndex to the appended item
	//item->setModelIndex(l_index);
	//// create modelIndices for childs of this item
	//for (int i=0; i<item->getNumberOfChilds(); i++)
	//{
	//	AbstractItem* l_pChild=item->getChild(i);
	//	QModelIndex l_childIndex=this->createIndex(i,1,l_pChild);
	//	l_pChild->setModelIndex(l_childIndex);
	//	// create modelIndices for childs of this child
	//	for (int j=0; j<l_pChild->getNumberOfChilds(); j++)
	//	{
	//		AbstractItem* l_pChildChild=l_pChild->getChild(j);
	//		QModelIndex l_childChildIndex=this->createIndex(j,2, l_pChildChild);
	//		l_pChildChild->setModelIndex(l_childChildIndex);
	//	}
	//}
	//// emit rowsInserted-event ??
	//endInsertRows();
	//item->setFocus(false);
	//// connect signals
	//bool test=connect(item, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItemData(const QModelIndex &,  const QModelIndex &)));
}

void SceneModel::clearModel(void) 
{
	m_data.clear(); 
	m_geomID=0;
	m_geomGroupNr=0;
	m_focusedItemIndex=QModelIndex();
	m_currentGeomGroupIndex=QModelIndex();
};

void SceneModel::removeItem(QModelIndex index)
{
	if (index.isValid())
	{
		// save parent item
		QModelIndex l_parentIndex=index.parent();
		AbstractItem* l_pParentItem=NULL;
		if (l_parentIndex.isValid())
			l_pParentItem=reinterpret_cast<AbstractItem*>(l_parentIndex.internalPointer());

		// get base item
		QModelIndex baseIndex=this->getBaseIndex(index);
		QModelIndex l_rootIndex;
		int removedGeomID;
		int geomGroupRemoveID;
		GeometryItem* l_pGeomToBeRemoved;
		AbstractItem* l_pRootItem;
		MiscItem* l_pMiscItem;
		GeomGroupItem* l_pGeomGroupItem;
		AbstractItem* l_pAbstract;
		GeometryItem* l_pGeom;
		// check wether base item is a geometry
		AbstractItem* l_pAbstractToBeRemoved=reinterpret_cast<AbstractItem*>(baseIndex.internalPointer());
		switch (l_pAbstractToBeRemoved->getObjectType())
		{
		case AbstractItem::GEOMETRY:
			l_pGeomToBeRemoved=reinterpret_cast<GeometryItem*>(l_pAbstractToBeRemoved);
			removedGeomID=l_pGeomToBeRemoved->getGeometryID();
			geomGroupRemoveID=l_pGeomToBeRemoved->getGeomGroupID();
			// adjust geomIDs of all geometries behind the geometry that that is about to be removed
			// delete geometry item from its respective geometryGroup
			l_rootIndex=this->getRootIndex(index);
			if (l_rootIndex.isValid())
			{
				l_pRootItem=reinterpret_cast<AbstractItem*>(l_rootIndex.internalPointer());
				if (l_pRootItem->getObjectType() == AbstractItem::MISCITEM)
				{
					l_pMiscItem=reinterpret_cast<MiscItem*>(l_pRootItem);
					if (l_pMiscItem->getMiscType()==MiscItem::GEOMETRYGROUP)
					{
						l_pGeomGroupItem=reinterpret_cast<GeomGroupItem*>(l_pMiscItem);
						l_pGeomGroupItem->removeChild(baseIndex.row());
						// decrement internal geometry counter
						m_geomID--;
					}
				}
				// adjust geomIDs of all geometries behind the geometry that that has just been removed
				// loop through all top level elements
				for (int i=0;i<m_data.size();i++)
				{
					l_pAbstract=reinterpret_cast<AbstractItem*>(m_data.at(i));
					// see if we have a geometryGroupItem
					if (l_pAbstract->getObjectType()==AbstractItem::MISCITEM)
					{
						l_pMiscItem=reinterpret_cast<MiscItem*>(l_pAbstract);
						if (l_pMiscItem->getMiscType()==MiscItem::GEOMETRYGROUP)
						{
							// if we have a geometryGroup, loop through all its geometries
							for (unsigned int iMisc=0; iMisc<l_pMiscItem->getNumberOfChilds(); iMisc++)
							{
								l_pGeom=reinterpret_cast<GeometryItem*>(l_pMiscItem->getChild(iMisc));
								if (l_pGeom != NULL)
								{
									// if the current geometryID is higher than that of the removed geometry, decrement it
									if (l_pGeom->getGeometryID() > l_pGeomToBeRemoved->getGeometryID())
										l_pGeom->setGeometryID(l_pGeom->getGeometryID()-1);
								}
							}
						}
					}
				}
			break;
		case AbstractItem::MISCITEM:
			l_pMiscItem=reinterpret_cast<MiscItem*>(l_pAbstractToBeRemoved);
			if (l_pMiscItem->getMiscType() == MiscItem::GEOMETRYGROUP)
			{
				this->m_geomGroupNr--;
				this->removeRow(baseIndex.row(),baseIndex.parent());
			}
			break;
		default:
			this->removeRow(baseIndex.row(),baseIndex.parent());
			break;
			}

		}

		// adjust model indices
		unsigned int l_numberOfChilds=0;
		if (l_pParentItem)
			l_numberOfChilds=l_pParentItem->getNumberOfChilds();
		else
			l_numberOfChilds=this->m_data.size();
		for (unsigned int i=index.row(); i< l_numberOfChilds; i++)
		{
			QModelIndex newIndex=this->index(i, 0, l_parentIndex);
			//QModelIndex newIndex=this->index(i, 0, l_pParentItem->getModelIndex());
			if (l_pParentItem)
				l_pParentItem->getChild(i)->setModelIndex(newIndex);
			else
				this->m_data.at(i)->setModelIndex(newIndex);
		}

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

// returns the modelIndex of the base item that ultimately holds the item with the given index. Base item is either the root item or in case of a geometry groups it is the geometry item
QModelIndex SceneModel::getBaseIndex(const QModelIndex &index)
{
	// if index already points to a geometry, smiply return index
	AbstractItem* l_pAbstractItem=reinterpret_cast<AbstractItem*>(index.internalPointer());
	if (l_pAbstractItem->getObjectType()==AbstractItem::GEOMETRY)
		return index;
	// if not, take a look at its parent
	else
	{
		QModelIndex parentIndex;
		parentIndex=parent(index);
		// if parent is invalid, we're already at root level and return the root index
		if (!parentIndex.isValid())
			return index;
		else
		{
			l_pAbstractItem=reinterpret_cast<AbstractItem*>(parentIndex.internalPointer());
			return getBaseIndex(parentIndex);
		}
	}
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
	QModelIndex baseItemIndex;
	if (index.isValid())
	{
		// find baseItem that ultimately holds the item that just changed
		baseItemIndex=this->getBaseIndex(index);
	}
	else
	{ 
		baseItemIndex=index;
	}
	if (baseItemIndex != m_focusedItemIndex)
	{
		if (baseItemIndex!=QModelIndex())
		{
			// if any item had focus before, remove focus from it now
			if (m_focusedItemIndex != QModelIndex())
				reinterpret_cast<AbstractItem*>(m_focusedItemIndex.internalPointer())->setFocus(false);

			// set new focus
			// all items in our model are AbstractItems, so we can savely cast here
			AbstractItem* l_pAbstractItem=this->getItem(baseItemIndex);
			l_pAbstractItem->setFocus(true);
		}
		m_focusedItemIndex=baseItemIndex;
		emit itemFocusChanged(index);
	}
}

void SceneModel::render(QMatrix4x4  &matrix, RenderOptions &options)
{
	for (int i=0; i < this->m_data.size(); i++)
		m_data.at(i)->render(matrix, options);
}

void SceneModel::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	for (int i=0; i < this->m_data.size(); i++)
		m_data.at(i)->renderVtk(renderer);
}

void SceneModel::updateVtk()
{
	for (int i=0; i < this->m_data.size(); i++)
		m_data.at(i)->updateVtk();
}

void SceneModel::setRenderOptions(RenderOptions options)
{
	m_renderOptions=options;
	for (int i=0; i < this->m_data.size(); i++)
		m_data.at(i)->setRenderOptions(options);
}