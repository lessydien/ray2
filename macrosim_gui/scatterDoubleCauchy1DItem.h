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

#ifndef SCATTERDOUBLECAUCHY1DITEM
#define SCATTERDOUBLECAUCHY1DITEM

#include <qicon.h>

#include "ScatterItem.h"

#include "QPropertyEditor/CustomTypes.h"


namespace macrosim 
{

/** @class ScatterItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScatterDoubleCauchy1DItem :
	public ScatterItem
{
	Q_OBJECT

	Q_PROPERTY(double m_gammaSl READ getGammaSl WRITE setGammaSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_gammaSp READ getGammaSp WRITE setGammaSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_kSl READ getKSl WRITE setKSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_kSp READ getKSp WRITE setKSp DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d M_scatAxis READ getScatAxis WRITE setScatAxis DESIGNABLE true USER true);
public:

	ScatterDoubleCauchy1DItem(QString name="DoubleCauchy1D", QObject *parent=0);
	~ScatterDoubleCauchy1DItem(void);

	double getGammaSl() const {return m_gammaSl;};
	void setGammaSl(const double in) {m_gammaSl=in; emit itemChanged(m_index, m_index);};
	double getGammaSp() const {return m_gammaSp;};
	void setGammaSp(const double in) {m_gammaSp=in; emit itemChanged(m_index, m_index);};
	double getKSl() const {return m_kSl;};
	void setKSl(const double in) {m_kSl=in; emit itemChanged(m_index, m_index);};
	double getKSp() const {return m_kSp;};
	void setKSp(const double in) {m_kSp=in; emit itemChanged(m_index, m_index);};
	Vec3d getScatAxis() const {return m_scatAxis;};
	void setScatAxis(const Vec3d in) {m_scatAxis=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;

	// functions for property editor

private:
	Vec3d m_scatAxis;
	double m_gammaSl;
	double m_gammaSp;
	double m_kSl;
	double m_kSp;

};

}; //namespace macrosim

#endif