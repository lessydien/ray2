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

#ifndef MATERIALLINEARGRATING1DITEM
#define MATERIALLINEARGRATING1DITEM

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
class MaterialLinearGrating1DItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(Vec3d diffAxis READ getDiffAxis WRITE setDiffAxis DESIGNABLE true USER true);
	Q_PROPERTY(Vec9si diffOrders READ getDiffOrders WRITE setDiffOrders DESIGNABLE true USER true);
	Q_PROPERTY(Vec9d diffEfficiencies READ getDiffEffs WRITE setDiffEffs DESIGNABLE true USER true);
	Q_PROPERTY(double gratingPeriod READ getGratingPeriod WRITE setGratingPeriod DESIGNABLE true USER true);
	Q_PROPERTY(QString diffFileName READ getDiffFileName WRITE setDiffFileName DESIGNABLE true USER true);
	Q_PROPERTY(double n1 READ getn1 WRITE setn1 DESIGNABLE true USER true);
	Q_PROPERTY(double n2 READ getn2 WRITE setn2 DESIGNABLE true USER true);
	Q_PROPERTY(QString glassName READ getGlassName WRITE setGlassName DESIGNABLE true USER true);
	Q_PROPERTY(QString immersionName READ getImmersionName WRITE setImmersionName DESIGNABLE true USER true);


public:
	
	MaterialLinearGrating1DItem(QString name="MaterialLinearGrating1D", QObject *parent=0);
	~MaterialLinearGrating1DItem(void);

	// functions for property editor
	void setDiffAxis(const Vec3d in) {m_diffAxis=in; emit itemChanged(m_index, m_index);};
	Vec3d getDiffAxis() const {return m_diffAxis;};
	void setDiffOrders(const Vec9si in) {m_diffOrders=in; emit itemChanged(m_index, m_index);};
	Vec9si getDiffOrders() const {return m_diffOrders;};
	void setDiffEffs(const Vec9d in) {m_diffEffs=in; emit itemChanged(m_index, m_index);};
	Vec9d getDiffEffs() const {return m_diffEffs;};
	void setn1(const double n) {m_n1=n; emit itemChanged(m_index, m_index);};
	double getn1() const {return m_n1;};
	void setn2(const double n) {m_n2=n; emit itemChanged(m_index, m_index);};
	double getn2() const {return m_n2;};
	void setGlassName(const QString in) {m_glassName=in; emit itemChanged(m_index, m_index);};
	QString getGlassName() const {return m_glassName;};
	void setImmersionName(const QString in) {m_immersionName=in; emit itemChanged(m_index, m_index);};
	QString getImmersionName() const {return m_immersionName;};
	void setDiffFileName(const QString in) {m_diffFileName=in; emit itemChanged(m_index, m_index);};
	QString getDiffFileName() const {return m_diffFileName;};
	void setGratingPeriod(const double in) {m_gratingPeriod=in; emit itemChanged(m_index, m_index);};
	double getGratingPeriod() const {return m_gratingPeriod;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	double getTransparency() {return 0.5;};

private:

	Vec3d m_diffAxis;
	Vec9si m_diffOrders;
	Vec9d m_diffEffs;
	double m_gratingPeriod;
	double m_n1;
	double m_n2;
	QString m_glassName;
	QString m_immersionName;
	QString m_diffFileName;
};

}; //namespace macrosim

#endif