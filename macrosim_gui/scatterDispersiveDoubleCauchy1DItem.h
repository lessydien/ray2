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

#ifndef SCATTERDISPDOUBLECAUCHY1D
#define SCATTERDISPDOUBLECAUCHY1D

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
class ScatterDispersiveDoubleCauchy1DItem :
	public ScatterItem
{
	Q_OBJECT

	Q_PROPERTY(double m_aGammaSl READ getAGammaSl WRITE setAGammaSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_aGammaSp READ getAGammaSp WRITE setAGammaSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_cGammaSl READ getCGammaSl WRITE setCGammaSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_cGammaSp READ getCGammaSp WRITE setCGammaSp DESIGNABLE true USER true);
	Q_PROPERTY(double m_aKSl READ getAKSl WRITE setAKSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_aKSp READ getAKSp WRITE setAKSp DESIGNABLE true USER true);
	Q_PROPERTY(double m_cKSl READ getCKSl WRITE setCKSl DESIGNABLE true USER true);
	Q_PROPERTY(double m_cKSp READ getCKSp WRITE setCKSp DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d M_scatAxis READ getScatAxis WRITE setScatAxis DESIGNABLE true USER true);

public:

	ScatterDispersiveDoubleCauchy1DItem(QString name="DispersiveDoubleCauchy1D", QObject *parent=0);
	~ScatterDispersiveDoubleCauchy1DItem(void);

	double getAGammaSl() const {return m_aGammaSl;};
	void setAGammaSl(const double in) {m_aGammaSl=in; emit itemChanged(m_index, m_index);};
	double getAGammaSp() const {return m_aGammaSp;};
	void setAGammaSp(const double in) {m_aGammaSp=in; emit itemChanged(m_index, m_index);};
	double getCGammaSl() const {return m_cGammaSl;};
	void setCGammaSl(const double in) {m_cGammaSl=in; emit itemChanged(m_index, m_index);};
	double getCGammaSp() const {return m_cGammaSp;};
	void setCGammaSp(const double in) {m_cGammaSp=in; emit itemChanged(m_index, m_index);};
	double getAKSl() const {return m_aKSl;};
	void setAKSl(const double in) {m_aKSl=in; emit itemChanged(m_index, m_index);};
	double getAKSp() const {return m_aKSp;};
	void setAKSp(const double in) {m_aKSp=in; emit itemChanged(m_index, m_index);};
	double getCKSl() const {return m_cKSl;};
	void setCKSl(const double in) {m_cKSl=in; emit itemChanged(m_index, m_index);};
	double getCKSp() const {return m_cKSp;};
	void setCKSp(const double in) {m_cKSp=in; emit itemChanged(m_index, m_index);};
	Vec3d getScatAxis() const {return m_scatAxis;};
	void setScatAxis(const Vec3d in) {m_scatAxis=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;

	// functions for property editor

private:
	double m_aGammaSl;
	double m_aGammaSp;
	double m_cGammaSl;
	double m_cGammaSp;
	double m_aKSl;
	double m_aKSp;
	double m_cKSl;
	double m_cKSp;
	Vec3d m_scatAxis;

};

}; //namespace macrosim

#endif