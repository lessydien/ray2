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

#ifndef COATINGNUMCOEFFSITEM
#define COATINGNUMCOEFFSITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "CoatingItem.h"


namespace macrosim 
{

/** @class CoatingItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class CoatingNumCoeffsItem :
	public CoatingItem
{
	Q_OBJECT

	Q_PROPERTY(double tA READ getAmplTransmittance WRITE setAmplTransmittance DESIGNABLE true USER true);
	Q_PROPERTY(double rA READ getAmplReflectance WRITE setAmplReflectance DESIGNABLE true USER true);


public:

	CoatingNumCoeffsItem(double rA=0, double tA=1, QString name="numCoeffs", QObject *parent=0);
	~CoatingNumCoeffsItem(void);

	// functions for property editor
	double getAmplTransmittance() const {return m_tA;};
	void setAmplTransmittance(const double tA) {m_tA=tA; emit itemChanged(m_index, m_index);};
	double getAmplReflectance() const {return m_rA;};
	void setAmplReflectance(const double rA) {m_rA=rA; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

private:

	double m_tA;
	double m_rA;
};

}; //namespace macrosim

#endif