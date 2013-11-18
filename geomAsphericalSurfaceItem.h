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

#ifndef ASPHERICALSURFACEITEM
#define ASPHERICALSURFACEITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>


using namespace macrosim;

namespace macrosim 
{

/** @class SphericalLenseItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class AsphericalSurfaceItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(double k READ getK WRITE setK DESIGNABLE true USER true);
	Q_PROPERTY(double c READ getC WRITE setC DESIGNABLE true USER true);
	Q_PROPERTY(double c2 READ getC2 WRITE setC2 DESIGNABLE true USER true);
	Q_PROPERTY(double c4 READ getC4 WRITE setC4 DESIGNABLE true USER true);
	Q_PROPERTY(double c6 READ getC6 WRITE setC6 DESIGNABLE true USER true);
	Q_PROPERTY(double c8 READ getC8 WRITE setC8 DESIGNABLE true USER true);
	Q_PROPERTY(double c10 READ getC10 WRITE setC10 DESIGNABLE true USER true);
	Q_PROPERTY(double c12 READ getC12 WRITE setC12 DESIGNABLE true USER true);
	Q_PROPERTY(double c14 READ getC14 WRITE setC14 DESIGNABLE true USER true);
	Q_PROPERTY(double c16 READ getC16 WRITE setC16 DESIGNABLE true USER true);
	Q_CLASSINFO("k", "decimals=10;");
	Q_CLASSINFO("c", "decimals=10;");
	Q_CLASSINFO("c2", "decimals=10;");
	Q_CLASSINFO("c4", "decimals=10;");
	Q_CLASSINFO("c6", "decimals=10;");
	Q_CLASSINFO("c8", "decimals=10;");
	Q_CLASSINFO("c10", "decimals=10;");
	Q_CLASSINFO("c12", "decimals=10;");
	Q_CLASSINFO("c14", "decimals=10;");
	Q_CLASSINFO("c16", "decimals=10;");
	//Q_PROPERTY(MaterialItem::MaterialType Material READ getMaterial WRITE setMaterial DESIGNABLE true USER true);


public:

	AsphericalSurfaceItem(QString name="AsphericalSurface", QObject *parent=0);
	~AsphericalSurfaceItem(void);

	// functions for property editor
	double getK() const {return m_k;};
	void setK(const double in) {m_k=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	void setC(const double in) {m_c=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC() const {return m_c;};
	void setC2(const double in) {m_c2=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC2() const {return m_c2;};
	void setC4(const double in) {m_c4=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC4() const {return m_c4;};
	void setC6(const double in) {m_c6=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC6() const {return m_c6;};
	void setC8(const double in) {m_c8=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC8() const {return m_c8;};
	void setC10(const double in) {m_c10=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC10() const {return m_c10;};
	void setC12(const double in) {m_c12=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC12() const {return m_c12;};
	void setC14(const double in) {m_c14=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC14() const {return m_c14;};
	void setC16(const double in) {m_c16=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getC16() const {return m_c16;};

	//Vec3f calcNormal(Vec3f vertex);

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	Vec3f calcNormal(Vec3f vertex);
	void render(QMatrix4x4 &m, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);
	
//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:
	void updateVtk();
	double calcZ(double r);

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;

//	MaterialItem::MaterialType m_materialType;
	double m_k; // concic constant
	double m_c; // 1/(radius of curvature)
	double m_c2;
	double m_c4;
	double m_c6;
	double m_c8;
	double m_c10;
	double m_c12;
	double m_c14;
	double m_c16;
}; // end class asphericalsurfaceitem

}; //namespace macrosim

#endif