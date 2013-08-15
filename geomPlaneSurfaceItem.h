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

#ifndef PLANESURFACEACEITEM
#define PLANESURFACEACEITEM

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
class PlaneSurfaceItem :
	public GeometryItem
{
	Q_OBJECT


public:

	PlaneSurfaceItem(QString name="PlaneSurface", QObject *parent=0);
	~PlaneSurfaceItem(void);

	// functions for property editor

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
};

}; //namespace macrosim

#endif