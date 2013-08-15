#ifndef DOCKWIDGETCONSOLE_H
	#define DOCKWIDGETCONSOLE_H

#include <qdockwidget.h>
#include <qstring.h>
#include <qplaintextedit.h>

class dockWidget_Console : 
	public QDockWidget
{
public:
	dockWidget_Console(const QString title=QString(), QWidget *parent=(QWidget*)0);
	~dockWidget_Console(void);

	void appendText(const QString in);

private:
	QPlainTextEdit *m_pTextEdit;

};


#endif