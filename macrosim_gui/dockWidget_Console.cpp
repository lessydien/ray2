#include "dockWidget_Console.h"


dockWidget_Console::dockWidget_Console(const QString title, QWidget *parent) :
	QDockWidget(title, parent),
		m_pTextEdit(NULL)
{
	m_pTextEdit = new QPlainTextEdit(this);
	m_pTextEdit->setReadOnly(true);
	this->setWidget(m_pTextEdit);
}


dockWidget_Console::~dockWidget_Console(void)
{
	delete m_pTextEdit;
	m_pTextEdit=NULL;
}

void dockWidget_Console::appendText(const QString in)
{
	m_pTextEdit->appendPlainText(in);
}
