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

#ifndef MATERIALIDEALLENSEITEM
#define MATERIALIDEALLENSEITEM

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
class MaterialIdealLenseItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(double f0 READ getF0 WRITE setF0 DESIGNABLE true USER true);
	Q_PROPERTY(double dispersionConstant READ getDispConst WRITE setDispConst DESIGNABLE true USER true);
	Q_PROPERTY(double lambda0 READ getLambda0 WRITE setLambda0 DESIGNABLE true USER true);

public:
	
	MaterialIdealLenseItem(double f0=0, double lambda0=0, double dispConst=0, QString name="MaterialIdealLense", QObject *parent=0);
	~MaterialIdealLenseItem(void);

	// functions for property editor
	void setF0(const double in) {m_f0=in; emit itemChanged(m_index, m_index);};
	double getF0() const {return m_f0;};
	void setLambda0(const double in) {m_lambda0=in; emit itemChanged(m_index, m_index);};
	double getLambda0() const {return m_lambda0;};
	void setDispConst(const double in) {m_dispConst=in; emit itemChanged(m_index, m_index);};
	double getDispConst() const {return m_dispConst;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);


private:

	double m_f0; // focal length at centre wavelength
	double m_dispConst; // dispersion constant 
	double m_lambda0; // centre wavelength
};

}; //namespace macrosim

#endif