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

#ifndef FIELDITEM
#define FIELDITEM

#include "DataObject/dataobj.h"

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "materialItem.h"

using namespace macrosim;

//namespace ito
//{
//class DataObject;
//}

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class FieldItem :
	public AbstractItem
{
	Q_OBJECT

	Q_PROPERTY(Vec3d tilt READ getTilt WRITE setTilt DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d root READ getRoot WRITE setRoot DESIGNABLE true USER true);
	Q_PROPERTY(FieldType fieldType READ getFieldType DESIGNABLE true USER true);
	Q_PROPERTY(double lambda READ getLambda WRITE setLambda DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d apertureHalfWidth READ getApertureHalfWidth WRITE setApertureHalfWidth DESIGNABLE true USER true);

	Q_ENUMS(FieldType);

public:

	enum FieldType {UNDEFINED, RAYFIELD, GEOMRAYFIELD, GEOMRAYFIELD_PSEUDOBANDWIDTH, DIFFRAYFIELD, DIFFRAYFIELDRAYAIM, PATHTRACERAYFIELD, INTENSITYFIELD, SCALARFIELD, VECFIELD, SCALARPLANEWAVE, SCALARSPHERICALWAVE, SCALARGAUSSIANWAVE, SCALARUSERWAVE, PATHINTTISSUERAYFIELD, GEOMRENDERFIELD};

	FieldItem(QString name="name", FieldType type=UNDEFINED, QObject *parent=0);
	~FieldItem(void);

	// functions for property editor
	const Vec3d getTilt()  {return m_tilt;};
	void setTilt(const Vec3d in) {m_tilt=in;};
	const FieldType getFieldType()  {return m_fieldType;};
	void setFieldType(const FieldType in) {m_fieldType=in;};
	double getLambda() const {return m_lambda;};
	void setLambda(const double in) {m_lambda=in;};
	Vec2d getApertureHalfWidth() const {return m_apertureHalfWidth;};
	void setApertureHalfWidth(const Vec2d in) {m_apertureHalfWidth=in;};
	Vec3d getRoot() const {return m_root;};
	void setRoot(const Vec3d root) {m_root=root;};

//	QString fieldTypeToString(const FieldType type) const;
//	FieldType stringToFieldType(const QString str) const;

	virtual bool signalDataChanged() {return true;};

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);
	virtual void render(QMatrix4x4 &m, RenderOptions &options) {};
	virtual void renderVtk(vtkSmartPointer<vtkRenderer> renderer) {};

	MaterialItem* getChild() const 
	{ 
		if (m_childs.empty())
			return NULL;
		else
		{
			if (m_childs[0]->getObjectType() != MATERIAL)
				return NULL;
			else
				return reinterpret_cast<MaterialItem*>(m_childs[0]);
		}
	};

	virtual void setChild(AbstractItem* child) 
	{
		if (child->getObjectType() == MATERIAL)
		{
			// so far we only allow one material per geometry. Therefore we clear the list if there are already entries present
			if (!m_childs.empty())
				m_childs.clear();

			m_childs.append(child);
		}
	}

protected:

	FieldType m_fieldType;
	double m_lambda;
	Vec2d m_apertureHalfWidth;
	Vec3d m_tilt;
	Vec3d m_root;
    
};

}; //namespace macrosim

#endif