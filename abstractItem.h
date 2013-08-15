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

#ifndef ABSTRACTITEM
#define ABSTRACTITEM

#include "GL/glew.h"
#include <qobject.h>
#include <qlist.h>
#include <QDomElement>
//#include <Qgraphicsitem>
//#include <QPainter>
#include <QModelIndex>
#include <QColor>
#include <qmatrix4x4.h>
#include "renderFuncs.h"
//#include "macrosim_scenemodel.h"

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkProperty.h>

namespace macrosim
{

class AbstractItem : public QObject
{
	Q_OBJECT

	Q_PROPERTY(QString Name READ getName WRITE setName DESIGNABLE true USER true);
	Q_PROPERTY(ObjectType Type READ getObjectType DESIGNABLE true USER true);
	Q_PROPERTY(Abstract_MaterialType materialType READ getMaterialType WRITE setMaterialType DESIGNABLE true USER true);
	Q_PROPERTY(bool render READ getRender WRITE setRender DESIGNABLE true USER true);	

	Q_ENUMS(ObjectType);
	Q_ENUMS(Abstract_MaterialType);

public:

	enum ObjectType {OBJUNKNOWN, FIELD, MISCITEM, GEOMETRY, MATERIAL, SCATTER, COATING, DETECTOR, GEOMETRYCONTAINER, FIELDCONTAINER, DETECTORCONTAINER, MISCCONTAINER};
	// note this has to be exactly the same definition including ordering as MaterialType in materialItem.h
	enum Abstract_MaterialType {MATUNKNOWN, REFRACTING, ABSORBING, DIFFRACTING, FILTER, LINGRAT1D, MATIDEALLENSE, REFLECTING, REFLECTINGCOVGLASS, PATHTRACESOURCE, DOE, VOLUMESCATTER};

	AbstractItem(ObjectType type=OBJUNKNOWN, QString name="name",  QObject *parent = 0) :
		QObject(parent),
		m_objectType(type),
		m_name(name),
		m_materialType(MATUNKNOWN),
		m_index(QModelIndex()),
		m_focus(false),
		m_render(true)
		{
			m_pActor = vtkSmartPointer<vtkActor>::New();
		}

	~AbstractItem(void) 
	{
		m_childs.clear();
	}

	virtual AbstractItem* getChild(unsigned int index) const
	{
		if(index >= m_childs.size()) return NULL;
		return m_childs[index];
	}
	int getNumberOfChilds() const { return m_childs.size(); };
	QString getName() const {return m_name;};
	void setName(const QString &name) {m_name=name; emit itemChanged(m_index, m_index);};
	ObjectType const getObjectType() const { return m_objectType; };
	void setObjectType(ObjectType type) { m_objectType=type; emit itemChanged(m_index, m_index);};
	Vec3d getRoot() const {return m_root;};
	void setRoot(const Vec3d root) {m_root=root; this->updateVtk(); emit itemChanged(m_index, m_index);};
//	virtual bool signalDataChanged() {return true;};
	void setModelIndex(QModelIndex &in) {m_index=in;};
	void createModelIndex(QModelIndex &parent, int row, int coloumn);
	QModelIndex getModelIndex() const {return m_index;};
	Abstract_MaterialType getMaterialType() const {return m_materialType;};
	void setMaterialType(const Abstract_MaterialType type);
	void setFocus(const bool in) {m_focus=in; this->updateVtk();};
	void setRender(const bool in);
	bool getRender() {return m_render;};
	vtkSmartPointer<vtkActor> getActor() {return m_pActor;};

	virtual double getTransparency() {return 1.0;};
	virtual QModelIndex hasActor(void *actor) const;

	virtual void removeFromView(vtkSmartPointer<vtkRenderer> renderer) {renderer->RemoveActor(m_pActor);};

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

	virtual void setChild(AbstractItem* child) 
	{
		m_childs.append(child);
		connect(child, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItem(const QModelIndex &, const QModelIndex &)));
	}

	virtual void removeChild(int index)
	{
		if ( (index>=0) && (index<m_childs.size()))
			m_childs.removeAt(index);
	}

	QString objectTypeToString(const ObjectType type) const
	{
		QString str;
		switch (type)
		{
		case FIELD:
			str= "FIELD";
			break;
		case MISCITEM:
			str= "MISCITEM";
			break;
		case GEOMETRY:
			str= "GEOMETRY";
			break;
		case MATERIAL:
			str= "MATERIAL";
			break;
		case SCATTER:
			str= "SCATTER";
			break;
		case COATING:
			str= "COATING";
			break;
		case DETECTOR:
			str="DETECTOR";
			break;
		default:
			str="OBJUNKNOWN";
			break;
		}
		return str;
	}

	ObjectType stringToObjectType(const QString str)
	{
		
		if (!str.compare("FIELD"))
			return FIELD;
		if (!str.compare("MISCITEM"))
			return MISCITEM;
		if (!str.compare("GEOMETRY"))
			return GEOMETRY;
		if (!str.compare("MATERIAL"))
			return MATERIAL;
		if (!str.compare("SCATTER"))
			return SCATTER;
		if (!str.compare("COATING"))
			return COATING;
		if (!str.compare("DETECTOR"))
			return DETECTOR;

		// error message
		return OBJUNKNOWN;
	}

	virtual void render(QMatrix4x4 &m, RenderOptions &options) {};
	virtual void renderVtk(vtkSmartPointer<vtkRenderer> renderer) {};
	virtual void updateVtk() {};
	virtual void setRenderOptions(RenderOptions options);

protected:
	QList<AbstractItem*> m_childs;
	ObjectType m_objectType;
	QString m_name;
	Vec3d m_root;
	QModelIndex m_index;
	Abstract_MaterialType m_materialType;
	bool m_focus;
	bool m_render;
	vtkSmartPointer<vtkActor> m_pActor;
	vtkSmartPointer<vtkMapper> m_pMapper;
	RenderOptions m_renderOptions;

private:

signals:
	void itemExchanged(const QModelIndex &parentIndex, const int row, const int coloumn, macrosim::AbstractItem &pNewItem);
	void itemChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	
	public slots:
		void changeItem(const QModelIndex &topLeft, const QModelIndex &bottomRight)	{ emit itemChanged(m_index, m_index); };

}; // end class AbstractItem

}; //end namespace

#endif