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

#ifndef SCATTERPHONGITEM
#define SCATTERPHONGITEM

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
class ScatterPhongItem :
	public ScatterItem
{
	Q_OBJECT

	Q_PROPERTY(double coefLambertian READ getCoefLambertian WRITE setCoefLambertian DESIGNABLE true USER true);
	Q_PROPERTY(double phongParam READ getPhongParam WRITE setPhongParam DESIGNABLE true USER true);
	Q_PROPERTY(double coefPhong READ getCoefPhong WRITE setCoefPhong DESIGNABLE true USER true);

public:

	ScatterPhongItem(double tis=1, QString name="Phong", QObject *parent=0);
	//ScatterPhongItem(double tis=1, QString name="Phong", double phongParam=0, double coefPhong=1);
	~ScatterPhongItem(void);

	// functions for property editor
	double getCoefLambertian() const {return m_coefLambertian;};
	void setCoefLambertian(const double tis) {m_coefLambertian=tis; emit itemChanged(m_index, m_index);};

	double getPhongParam() const {return m_phongParam;};
	void setPhongParam(const double phongParam) {m_phongParam=phongParam; emit itemChanged(m_index, m_index);};

	double getCoefPhong() const {return m_coefPhong;};
	void setCoefPhong(const double coefPhong) {m_coefPhong=coefPhong; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const ;
    bool readFromXML(const QDomElement &node);

private:

	double m_coefLambertian;
	double m_phongParam;
	double m_coefPhong;
};

}; //namespace macrosim

#endif