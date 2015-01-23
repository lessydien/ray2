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

#ifndef SCATTERCOOKTORRANCEITEM
#define SCATTERCOOKTORRANCEITEM

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
class ScatterCookTorranceItem :
	public ScatterItem
{
	Q_OBJECT

	Q_PROPERTY(double coefLambertian READ getCoefLambertian WRITE setCoefLambertian DESIGNABLE true USER true);
	Q_PROPERTY(double fresnelParam READ getFresnelParam WRITE setFresnelParam DESIGNABLE true USER true);
	Q_PROPERTY(double roughnessFactor READ getRoughnessFactor WRITE setRoughnessFactor DESIGNABLE true USER true);

public:

	ScatterCookTorranceItem(double tis=1, QString name="CookTorrance", QObject *parent=0);
	~ScatterCookTorranceItem(void);

	// functions for property editor
	double getCoefLambertian() const {return m_coefLambertian;};
	void setCoefLambertian(const double tis) {m_coefLambertian=tis; emit itemChanged(m_index, m_index);};

	double getFresnelParam() const {return m_fresnelParam;};
	void setFresnelParam(const double fresnelParam) {m_fresnelParam=fresnelParam; emit itemChanged(m_index, m_index);};

	double getRoughnessFactor() const {return m_roughnessFactor;};
	void setRoughnessFactor(const double roughnessFactor) {m_roughnessFactor=roughnessFactor; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const ;
    bool readFromXML(const QDomElement &node);

private:

	double m_coefLambertian;
	double m_fresnelParam;
	double m_roughnessFactor;
};

}; //namespace macrosim

#endif