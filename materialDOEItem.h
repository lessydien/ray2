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

#ifndef MATERIALDOEITEM
#define MATERIALDOEITEM

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
class MaterialDOEItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(double n1 READ getn1 WRITE setn1 DESIGNABLE true USER true);
	Q_PROPERTY(double n2 READ getn2 WRITE setn2 DESIGNABLE true USER true);
	Q_PROPERTY(double stepHeight READ getStepHeight WRITE setStepHeight DESIGNABLE true USER true);
	Q_PROPERTY(int DOEnr READ getDOEnr WRITE setDOEnr DESIGNABLE true USER true);
	Q_PROPERTY(QString glassName READ getGlassName WRITE setGlassName DESIGNABLE true USER true);
	Q_PROPERTY(QString immersionName READ getImmersionName WRITE setImmersionName DESIGNABLE true USER true);

	Q_PROPERTY(QString DOEFilename READ getFilename WRITE setFilename DESIGNABLE true USER true);
	Q_PROPERTY(QString DOE_Effs_baseFilename READ getDOEBaseFilename WRITE setDOEBaseFilename DESIGNABLE true USER true);

public:
	
	MaterialDOEItem(double n1=1, double n2=1, QString name="MaterialDOE", QObject *parent=0);
	~MaterialDOEItem(void);

	// functions for property editor
	void setn1(const double n) {m_n1=n; emit itemChanged(m_index, m_index);};
	double getn1() const {return m_n1;};
	void setn2(const double n) {m_n2=n; emit itemChanged(m_index, m_index);};
	double getn2() const {return m_n2;};
	void setStepHeight(const double in) {m_stepHeight=in; emit itemChanged(m_index, m_index);};
	double getStepHeight() const {return m_stepHeight;};
	void setDOEnr(const int in) {m_dOEnr=in; emit itemChanged(m_index, m_index);};
	int getDOEnr() const {return m_dOEnr;};
	void setGlassName(const QString in) {m_glassName=in; emit itemChanged(m_index, m_index);};
	QString getGlassName() const {return m_glassName;};
	void setImmersionName(const QString in) {m_immersionName=in; emit itemChanged(m_index, m_index);};
	QString getImmersionName() const {return m_immersionName;};
	QString getFilename() const {return m_filenameDOE;};
	void setFilename(const QString in) {m_filenameDOE=in; emit itemChanged(m_index, m_index);};
	QString getDOEBaseFilename() const {return m_filenameBaseDOEeffs;};
	void setDOEBaseFilename(const QString in) {m_filenameBaseDOEeffs=in; emit itemChanged(m_index, m_index);};


	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);


private:

	double m_n1;
	double m_n2;
	double m_stepHeight;
	int m_dOEnr;
	QString m_glassName;
	QString m_immersionName;
	QString m_filenameDOE;
	QString m_filenameBaseDOEeffs;
};

}; //namespace macrosim

#endif