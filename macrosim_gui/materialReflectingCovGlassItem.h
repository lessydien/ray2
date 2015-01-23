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

#ifndef MATERIALREFLECTINGCOVGLASSITEM
#define MATERIALREFLECTINGCOVGLASSITEM

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
class MaterialReflectingCovGlassItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(double tA READ getAmplTransmittance WRITE setAmplReflectance DESIGNABLE true USER true);
	Q_PROPERTY(double rA READ getAmplReflectance WRITE setAmplReflectance DESIGNABLE true USER true);
	Q_PROPERTY(int geometryID READ getGeometryID WRITE setGeometryID DESIGNABLE true USER true);

public:
	
	MaterialReflectingCovGlassItem(double rA=0, double tA=1, int geometryID=0, QString name="MaterialReflectingCovGlass", QObject *parent=0);
	~MaterialReflectingCovGlassItem(void);

	// functions for property editor
	void setAmplTransmittance(const double in) {m_tA=in; emit itemChanged(m_index, m_index);};
	double getAmplTransmittance() const {return m_tA;};
	void setAmplReflectance(const double in) {m_rA=in; emit itemChanged(m_index, m_index);};
	double getAmplReflectance() const {return m_rA;};
	void setGeometryID(const double in) {m_geometryID=in; emit itemChanged(m_index, m_index);};
	double getGeometryID() const {return m_geometryID;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);


private:

	double m_rA;
	double m_tA;
	int m_geometryID;
};

}; //namespace macrosim

#endif