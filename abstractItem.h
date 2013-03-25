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

#include <qobject.h>
#include <qlist.h>
#include <QDomElement>
#include <Qgraphicsitem>
#include <QPainter>
#include <QModelIndex>
#include <QColor>
#include <qmatrix4x4.h>
#include "renderFuncs.h"

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

	enum ObjectType {OBJUNKNOWN, FIELD, GEOMETRYGROUP, GEOMETRY, MATERIAL, SCATTER, COATING, DETECTOR, GEOMETRYCONTAINER, FIELDCONTAINER, DETECTORCONTAINER};
	// note this has to be exactly the same definition including ordering as MaterialType in materialItem.h
	enum Abstract_MaterialType {MATUNKNOWN, REFRACTING, ABSORBING, DIFFRACTING, FILTER, LINGRAT1D, MATIDEALLENSE, REFLECTING, REFLECTINGCOVGLASS, PATHTRACESOURCE, DOE};

	AbstractItem(ObjectType type=OBJUNKNOWN, QString name="name",  QObject *parent = 0) :
		QObject(parent),
		m_objectType(type),
		m_name(name),
		m_materialType(MATUNKNOWN),
		m_index(QModelIndex()),
		m_focus(false),
		m_render(true)
		{

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
//	virtual bool signalDataChanged() {return true;};
	void setModelIndex(QModelIndex &in) {m_index=in;};
	QModelIndex getModelIndex() const {return m_index;};
	void setRoot(const Vec3d &in) {m_root=in; emit itemChanged(m_index, m_index);};
	Vec3d getRoot() const {return m_root;};
	Abstract_MaterialType getMaterialType() const {return m_materialType;};
	void setMaterialType(const Abstract_MaterialType type);
	void setFocus(const bool in) {m_focus=in;};
	void setRender(const bool in) {m_render=in;};
	bool getRender() {return m_render;};


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
		case GEOMETRYGROUP:
			str= "GEOMETRYGROUP";
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
		if (!str.compare("GEOMETRYGROUP"))
			return GEOMETRYGROUP;
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

protected:
	QList<AbstractItem*> m_childs;
	ObjectType m_objectType;
	QString m_name;
	Vec3d m_root;
	QModelIndex m_index;
	Abstract_MaterialType m_materialType;
	bool m_focus;
	bool m_render;

private:

public:
	virtual void render(QMatrix4x4 &m, RenderOptions &options);

signals:
	void itemChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

	public slots:
		void changeItem(const QModelIndex &topLeft, const QModelIndex &bottomRight)	{ emit itemChanged(m_index, m_index); };

}; // end class AbstractItem

}; //end namespace

#endif