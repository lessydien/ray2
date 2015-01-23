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

#ifndef GEOMCADOBJECTITEM
#define GEOMCADOBJECTITEM

#include <nvModel.h>

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"
#include <vtkOBJReader.h>

using namespace macrosim;

namespace macrosim 
{

/** @class CadObjectItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class CadObjectItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(QString ObjFilename READ getObjFilename WRITE setObjFilename DESIGNABLE true USER true);

public:

	CadObjectItem(QString name="CadObject", QObject *parent=0);
	~CadObjectItem(void);

	// functions for property editor
	QString getObjFilename() const {return m_objFilename;};
	void setObjFilename(const QString in);

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);
	void updateVtk();

//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:

	nv::Model* model;
	GLuint modelVB;
	GLuint modelIB;
	nv::vec3f modelBBMin, modelBBMax, modelBBCenter;

	vtkSmartPointer<vtkOBJReader> m_pReader;

	QString m_objFilename;
	bool m_objFileLoaded;
	bool m_objFileChanged;
};

}; //namespace macrosim

#endif