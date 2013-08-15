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

#ifndef COATINGITEM
#define COATINGITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"


namespace macrosim 
{

/** @class CoatingItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class CoatingItem :
	public AbstractItem
{
	Q_OBJECT

	Q_ENUMS(CoatingType);

	Q_PROPERTY(CoatingType CoatType READ getCoatingType DESIGNABLE true USER true);
	Q_PROPERTY(Mat_CoatingType CoatType DESIGNABLE true USER true); // overwrite coatingType-Property of materialItem, so it can not be changed in propertyEditor of this item
	Q_PROPERTY(Abstract_MaterialType materialType DESIGNABLE true USER true); // overwrite materialType-Property of abstractItem, so it can not be changed in propertyEditor of this item
	


public:
	enum CoatingType {NOCOATING, NUMCOEFFS, FRESNELCOEFFS, DISPNUMCOEFFS};

	CoatingItem(CoatingType CoatType=NOCOATING, QString name="base coating", QObject *parent=0);
	~CoatingItem(void);

	// functions for property editor
	CoatingType getCoatingType() const {return m_coatingType;};
	void setCoatingType(const CoatingType type) {m_coatingType=type; emit itemChanged(m_index, m_index);};

	//QString coatingTypeToString(const CoatingType type) const;
	//CoatingType stringToCoatingType(const QString str) const;

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

private:

	CoatingType m_coatingType;

signals:
	void itemChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

public slots:
	void changeItem(const QModelIndex &topLeft, const QModelIndex & bottomRight);

};

}; //namespace macrosim

#endif