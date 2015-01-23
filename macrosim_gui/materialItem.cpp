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

#include "MaterialItem.h"
#include "materialItemLib.h"
#include "coatingItemLib.h"
#include "scatterItemLib.h"
//#include "CoatingNoItem.h"
//#include "ScatterNoItem.h"

using namespace macrosim;

MaterialItem::MaterialItem(MaterialType type, QString name, QObject *parent) :
	AbstractItem(MATERIAL, name, parent),
		m_materialType(type)
{
	// init coating to NOCOATING
	m_childs.append(new CoatingNoItem());
	m_coatType=NOCOATING;
	// init scatter to NOSCATTER
	m_childs.append(new ScatterNoItem());
	m_scatType=NOSCATTER;
}


MaterialItem::~MaterialItem()
{
	m_childs.clear();
}

ScatterItem* MaterialItem::getScatter() const
{
	for (unsigned int i=0; i<m_childs.count(); i++)
	{
		if (m_childs[i]->getObjectType() == SCATTER)
			return reinterpret_cast<ScatterItem*>(m_childs[i]);
	}
	return NULL;
}

CoatingItem* MaterialItem::getCoating() const
{
	for (unsigned int i=0; i<m_childs.count(); i++)
	{
		if (m_childs[i]->getObjectType() == COATING)
			return reinterpret_cast<CoatingItem*>(m_childs[i]);
	}
	return NULL;
}

void MaterialItem::setScatType(const Mat_ScatterType type)
{
	if (m_scatType != type)
	{
		m_scatType=type;
		ScatterItemLib l_scatLib;
		ScatterItem::ScatterType l_newScatType=l_scatLib.matScatTypeToScatType(type);
		ScatterItem* l_pNewScatter=l_scatLib.createScatter(l_newScatType);
		AbstractItem* l_pOldScatter=this->getChild(1);
		QModelIndex l_oldParentIndex;
		if (l_pOldScatter!=NULL)
		{
			l_oldParentIndex=this->getModelIndex();
			this->m_childs.replace(1, l_pNewScatter);
			emit itemExchanged(l_oldParentIndex, 1, 1, *l_pNewScatter);
		}
//		this->setChild(l_pNewScatter);
		emit itemChanged(m_index, m_index);
	}
}

void MaterialItem::setCoatType(const Mat_CoatingType type)
{
	if (m_coatType != type)
	{
		m_coatType=type;
		CoatingItemLib l_coatLib;
		CoatingItem::CoatingType l_newCoatType=l_coatLib.matCoatTypeToCoatType(type);
		CoatingItem* l_pNewCoat=l_coatLib.createCoating(l_newCoatType);
		AbstractItem* l_pOldCoating=this->getChild(0);
		QModelIndex l_oldParentIndex;
		if (l_pOldCoating != NULL)
		{
			l_oldParentIndex=this->getModelIndex();
			this->m_childs.replace(0, l_pNewCoat);
			emit itemExchanged(l_oldParentIndex, 0, 0, *l_pNewCoat);
		}
		//this->setChild(l_pNewCoat);
		emit itemChanged(m_index, m_index);
	}
}

bool MaterialItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	if (!AbstractItem::writeToXML(document, root))
		return false;
	// write scatter
	ScatterItem* l_pScatter=this->getScatter();
	if (l_pScatter != NULL)
		l_pScatter->writeToXML(document, root);
	CoatingItem* l_pCoat=this->getCoating();
	// write coating
	if (l_pCoat != NULL)
		l_pCoat->writeToXML(document, root);
	return true;
}

bool MaterialItem::readFromXML(const QDomElement &node)
{
//	MaterialItemLib l_materialItemLib;
//	m_materialType=l_materialItemLib.stringToMaterialType(node.attribute("materialType"));

	// look for scatter
	QDomNodeList l_scatNodeList=node.elementsByTagName("scatter");
	QDomElement l_scatElementXML;
	if (l_scatNodeList.count() == 0)
		l_scatElementXML=QDomElement();
	else
		l_scatElementXML=l_scatNodeList.at(0).toElement();
	ScatterItemLib l_scatterLib;
	ScatterItem l_scatterItem;
	QString l_scatTypeStr=l_scatElementXML.attribute("scatterType");
	ScatterItem* l_pScatter = l_scatterLib.createScatter(l_scatterLib.stringToScatterType(l_scatTypeStr));
	if (!l_pScatter->readFromXML(l_scatElementXML))
		return false;

	MaterialItemLib l_matItemLib;
	m_scatType=l_matItemLib.stringToMatScatterType(l_scatTypeStr);
	
	// look for coating
	QDomNodeList l_coatNodeList=node.elementsByTagName("coating");
	QDomElement l_coatElementXML;
	if (l_coatNodeList.count() == 0)
		l_coatElementXML=QDomElement();
	else
		l_coatElementXML=l_coatNodeList.at(0).toElement();

	CoatingItemLib l_coatingLib;
	CoatingItem l_coatingItem;
	QString l_coatTypeStr;
	l_coatTypeStr=l_coatElementXML.attribute("coatingType");
	CoatingItem* l_pCoating = l_coatingLib.createCoating(l_coatingLib.stringToCoatingType(l_coatTypeStr));
	if (!l_pCoating->readFromXML(l_coatElementXML))
		return false;

	m_coatType=l_matItemLib.stringToMatCoatingType(l_coatTypeStr);

	this->setChild(l_pCoating);
	this->setChild(l_pScatter);

	return true;
}

bool MaterialItem::signalDataChanged()
{
	ScatterItem *l_pScatterItem=reinterpret_cast<ScatterItem*>(this->getScatter());
	// if scatter changed, we replace the current scatter with the new one
	if (this->getScatType() != l_pScatterItem->getScatterType())
	{
		ScatterItemLib l_scatItemLib;
		MaterialItemLib l_matItemLib;
		// create scatterType from Mat_ScatterType (this is ugly... )
		QString str=l_matItemLib.matScatterTypeToString(this->getScatType());
		ScatterItem::ScatterType l_scatType=l_scatItemLib.stringToScatterType(str);
		this->setChild(l_scatItemLib.createScatter(l_scatType));
		return true;
	}
	CoatingItem *l_pCoatingItem=reinterpret_cast<CoatingItem*>(this->getCoating());
	// if coating changed, we replace the current coating with the new one
	if (this->getCoatType() != l_pCoatingItem->getCoatingType())
	{
		CoatingItemLib l_coatItemLib;
		MaterialItemLib l_matItemLib;
		// create coatingType from Mat_CoatingType (this is ugly... )
		QString str=l_matItemLib.matCoatingTypeToString(this->getCoatType());
		CoatingItem::CoatingType l_coatType=l_coatItemLib.stringToCoatingType(str);
		this->setChild(l_coatItemLib.createCoating(l_coatType));
		return true;
	}
	return true;
}

AbstractItem::Abstract_MaterialType MaterialItem::materialTypeToAbstractMaterialType(MaterialType in) 
{
    switch (in)
    {
    case MaterialItem::MATUNKNOWN:
        return AbstractItem::MATUNKNOWN;
        break;
    case MaterialItem::REFRACTING:
        return AbstractItem::REFRACTING;
        break;
    case MaterialItem::ABSORBING:
        return AbstractItem::ABSORBING;
        break;
    case MaterialItem::DIFFRACTING:
        return AbstractItem::DIFFRACTING;
        break;
    case MaterialItem::FILTER:
        return AbstractItem::FILTER;
        break;
    case MaterialItem::LINGRAT1D:
        return AbstractItem::LINGRAT1D;
        break;
    case MaterialItem::MATIDEALLENSE:
        return AbstractItem::MATIDEALLENSE;
        break;
    case MaterialItem::REFLECTING:
        return AbstractItem::REFLECTING;
        break;
    case MaterialItem::REFLECTINGCOVGLASS:
        return AbstractItem::REFLECTINGCOVGLASS;
        break;
    case MaterialItem::PATHTRACESOURCE:
        return AbstractItem::PATHTRACESOURCE;
        break;
    case MaterialItem::DOE:
        return AbstractItem::DOE;
        break;
    case MaterialItem::VOLUMESCATTER:
        return AbstractItem::VOLUMESCATTER;
        break;
    case MaterialItem::VOLUMEABSORBING:
        return AbstractItem::VOLUMEABSORBING;
        break;
    case MaterialItem::RENDERLIGHT:
        return AbstractItem::RENDERLIGHT;
        break;
    case MaterialItem::RENDERFRINGEPROJ:
        return AbstractItem::RENDERFRINGEPROJ;
        break;
    default:
        return AbstractItem::MATUNKNOWN;
        break;
    }
}


