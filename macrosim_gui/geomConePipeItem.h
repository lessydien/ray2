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

#ifndef CONEPIPEITEM
#define CONEPIPEITEM

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
class ConePipeItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(Vec2d apertureRadius2 READ getApertureRadius2 WRITE setApertureRadius2 DESIGNABLE true USER true);
	Q_PROPERTY(double thickness READ getThickness WRITE setThickness DESIGNABLE true USER true);
	Q_CLASSINFO("apertureRadius2", "decimals=10;");
	Q_CLASSINFO("thickness", "decimals=10;");

	//Q_PROPERTY(MaterialItem::MaterialType Material READ getMaterial WRITE setMaterial DESIGNABLE true USER true);


public:

	ConePipeItem(QString name="ConePipe", QObject *parent=0);
	~ConePipeItem(void);

	// functions for property editor
	Vec2d getApertureRadius2() const {return m_apertureRadius2;};
	void setApertureRadius2(const Vec2d in) {m_apertureRadius2=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getThickness() const {return m_thickness;};
	void setThickness(const double in) {m_thickness=in; this->updateVtk(); emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);
	Vec3f calcNormal(Vec3f vertex);
	
//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:
	void updateVtk();

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;

//	MaterialItem::MaterialType m_materialType;
	Vec2d m_apertureRadius2; // radius of end face
	double m_thickness; // length of cone segment

};

}; //namespace macrosim

#endif