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

#ifndef SCATTERTORRSPARR2DPATHTRACEITEM
#define SCATTERTORRSPARR2DPATHTRACEITEM

#include <qicon.h>

#include "scatterItem.h"

#include "QPropertyEditor/CustomTypes.h"


namespace macrosim 
{

/** @class ScatterItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScatterTorrSparr2DPathTraceItem :
	public ScatterItem
{
	Q_OBJECT

	Q_PROPERTY(double kDl READ getKDl WRITE setKDl DESIGNABLE true USER true);
	Q_PROPERTY(double kSl READ getKSl WRITE setKSl DESIGNABLE true USER true);
	Q_PROPERTY(double kSp READ getKDl WRITE setKSp DESIGNABLE true USER true);
	Q_PROPERTY(double kSigmaSl READ getSigmaSl WRITE setSigmaSl DESIGNABLE true USER true);
	Q_PROPERTY(double kSigmaSp READ getSigmaSp WRITE setSigmaSp DESIGNABLE true USER true);

public:

	ScatterTorrSparr2DPathTraceItem(QString name="TorrSparr2DPathTrace", QObject *parent=0);
	~ScatterTorrSparr2DPathTraceItem(void);

	double getKDl() const {return m_kDl;};
	void setKDl(const double in) {m_kDl=in; emit itemChanged(m_index, m_index);};
	double getKSl() const {return m_kSl;};
	void setKSl(const double in) {m_kSl=in; emit itemChanged(m_index, m_index);};
	double getKSp() const {return m_kSp;};
	void setKSp(const double in) {m_kSp=in; emit itemChanged(m_index, m_index);};
	double getSigmaSl() const {return m_sigmaSl;};
	void setSigmaSl(const double in) {m_sigmaSl=in; emit itemChanged(m_index, m_index);};
	double getSigmaSp() const {return m_sigmaSp;};
	void setSigmaSp(const double in) {m_sigmaSp=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;

	// functions for property editor

private:

	double m_kDl; // proportionality factor of diffuse lobe
	double m_kSl; // proportionality factor of specular lobe
	double m_kSp; // proportionality factor of specular peak
	double m_sigmaSl; // sigma of specular lobe
	double m_sigmaSp; // sigma of specular peak
};

}; //namespace macrosim

#endif