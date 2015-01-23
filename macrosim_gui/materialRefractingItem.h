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

#ifndef MATERIALREFRACTINGITEM
#define MATERIALREFRACTINGITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "GeometryItem.h"
#include "MaterialItem.h"


namespace macrosim 
{

/** @class MaterialItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MaterialRefractingItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(double n1 READ getn1 WRITE setn1 DESIGNABLE true USER true);
	Q_PROPERTY(double n2 READ getn2 WRITE setn2 DESIGNABLE true USER true);
	Q_PROPERTY(QString glassName READ getGlassName WRITE setGlassName DESIGNABLE true USER true);
	Q_PROPERTY(QString immersionName READ getImmersionName WRITE setImmersionName DESIGNABLE true USER true);

public:
	
	MaterialRefractingItem(double n1=1, double n2=1, QString name="MaterialRefracting", QObject *parent=0);
	~MaterialRefractingItem(void);

	// functions for property editor
	void setn1(const double n) {m_n1=n; emit itemChanged(m_index, m_index);};
	double getn1() const {return m_n1;};
	void setn2(const double n) {m_n2=n; emit itemChanged(m_index, m_index);};
	double getn2() const {return m_n2;};
	void setGlassName(const QString in) {m_glassName=in; emit itemChanged(m_index, m_index);};
	QString getGlassName() const {return m_glassName;};
	void setImmersionName(const QString in) {m_immersionName=in; emit itemChanged(m_index, m_index);};
	QString getImmersionName() const {return m_immersionName;};


	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	double getTransparency() {return 0.5;};


private:

	double m_n1;
	double m_n2;
	QString m_glassName;
	QString m_immersionName;
};

}; //namespace macrosim

#endif