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

#ifndef MATERIALDIFFRACTINGITEM
#define MATERIALDIFFRACTINGITEM

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
class MaterialDiffractingItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(double n1 READ getn1 WRITE setn1 DESIGNABLE true USER true);
	Q_PROPERTY(double n2 READ getn2 WRITE setn2 DESIGNABLE true USER true);

public:
	
	MaterialDiffractingItem(double n1=1, double n2=1, QString name="MaterialDiffracting", QObject *parent=0);
	~MaterialDiffractingItem(void);

	// functions for property editor
	void setn1(const double n) {m_n1=n; emit itemChanged(m_index, m_index);};
	double getn1() const {return m_n1;};
	void setn2(const double n) {m_n2=n; emit itemChanged(m_index, m_index);};
	double getn2() const {return m_n2;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);


private:

	double m_n1;
	double m_n2;
};

}; //namespace macrosim

#endif