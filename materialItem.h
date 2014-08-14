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

#ifndef MATERIALITEM
#define MATERIALITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "ScatterItem.h"
#include "CoatingItem.h"


namespace macrosim 
{

/** @class MaterialItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MaterialItem :
	public AbstractItem
{
	Q_OBJECT

	Q_ENUMS(MaterialType);
	Q_ENUMS(Mat_ScatterType);
	Q_ENUMS(Mat_CoatingType);

	//Q_PROPERTY(MaterialType MatType READ getMaterialType DESIGNABLE false USER false);
	Q_PROPERTY(Abstract_MaterialType materialType READ getAbstractMaterialType DESIGNABLE true USER true); // overwrite materialType-Property of abstractItem, so it can not be changed in propertyEditor of this item
	Q_PROPERTY(Mat_ScatterType ScatType READ getScatType WRITE setScatType DESIGNABLE true USER true);
	Q_PROPERTY(Mat_CoatingType CoatType READ getCoatType WRITE setCoatType DESIGNABLE true USER true);
	

public:
	// note this has to be exactly the same definition including ordering as AbstractMaterialType in abstractItem.h
	enum MaterialType {MATUNKNOWN, REFRACTING, ABSORBING, DIFFRACTING, FILTER, LINGRAT1D, MATIDEALLENSE, REFLECTING, REFLECTINGCOVGLASS, PATHTRACESOURCE, DOE, VOLUMESCATTER, VOLUMEABSORBING, RENDERLIGHT, RENDERFRINGEPROJ};
	enum Mat_ScatterType {NOSCATTER, LAMBERT2D, TORRSPARR1D, TORRSPARR2D, TORRSPARR2DPATHTRACE, DISPDOUBLECAUCHY1D, DOUBLECAUCHY1D, PHONG};
	enum Mat_CoatingType {NOCOATING, NUMCOEFFS};
	enum Test {TEST1, TEST2};

	MaterialItem(MaterialType type=ABSORBING, QString name="name", QObject *parent=0);
	~MaterialItem(void);

	bool signalDataChanged();

	// functions for property editor
	MaterialType getMaterialType() const {return m_materialType;};
	void setMaterialType(const MaterialType type) {m_materialType=type; emit itemChanged(m_index, m_index);};
    Abstract_MaterialType getAbstractMaterialType()  {return materialTypeToAbstractMaterialType(m_materialType);};
	Mat_ScatterType getScatType() const {return m_scatType;};
	void setScatType(const Mat_ScatterType type);
	Mat_CoatingType getCoatType() const {return m_coatType;};
	void setCoatType(const Mat_CoatingType type);

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

	//MaterialType stringToMaterialType(const QString str) const;
	//QString materialTypeToString(const MaterialType type) const;
	//Mat_ScatterType stringToMatScatterType(const QString str) const;
	//QString matScatterTypeToString(const Mat_ScatterType type) const;
	//Mat_CoatingType stringToMatCoatingType(const QString str) const;
	//QString matCoatingTypeToString(const Mat_CoatingType type) const;


	ScatterItem* getScatter() const;
	CoatingItem* getCoating() const;

	virtual void setChild(AbstractItem* child) 
	{
		if (child->getObjectType() == SCATTER)
		{
			bool replaced=false;
//			for (int index=0; index<m_childs.size(); index++)
//			{
			if (m_childs.size()>1)
			{
//				if (m_childs[index]->getObjectType() == SCATTER)
//				{
					m_childs.replace(1, child);
//					m_childs.replace(index, child);
					replaced=true;
//				}
			}
			if (!replaced)
				m_childs.append(child);
		}
		if (child->getObjectType() == COATING)
		{
			bool replaced=false;
//			for (int index=0; index<m_childs.size(); index++)
//			{
			if (m_childs.size()>0)
			{
//				if (m_childs[index]->getObjectType() == COATING)
//				{
					//m_childs.replace(index, child);
					m_childs.replace(0, child);
					replaced=true;
//				}
			}
			if (!replaced)
				m_childs.append(child);
		}
		connect(child, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItem(const QModelIndex &, const QModelIndex &)));
	}



private:
    Abstract_MaterialType materialTypeToAbstractMaterialType(MaterialType in);

	MaterialType m_materialType;
	Mat_ScatterType m_scatType;
	Mat_CoatingType m_coatType;

signals:
	void itemChanged(QModelIndex &topLeft, QModelIndex &bottomRight);

public slots:
	void changeItem(QModelIndex &topLeft, QModelIndex &bottomRight) {emit itemChanged(m_index, m_index);};

};

}; //namespace macrosim

#endif