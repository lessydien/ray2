#include "consoleStream.h"

//! constructor
/*!
    initializes this instance and stores actual content of stream in m_old_buf

    \param stream Stream of type std::ostream which should be observed
    \param type message type of enumeration tMsgType which corresponds to the stream
    \param lineBreak string representation of line break, default: \n
    \return description
    \sa tMsgType
*/
ConsoleStream::ConsoleStream(std::ostream &stream, QString lineBreak) :
	m_stream(stream)
{
    m_old_buf = stream.rdbuf();
    stream.rdbuf(this);
    line_break = lineBreak;
}

//! destructor
/*!
    destroys this instance and the stream observation and emits remaining string in the buffer.
    Restores m_old_buf to the stream.
*/
ConsoleStream::~ConsoleStream(void)
{
    // output anything that is left
    if (!m_string.empty())
    {
        emit flushStream(QString(m_string.c_str()));
    }
    m_stream.rdbuf(m_old_buf);
}

//! method invoked if new content has been added to stream
std::streamsize ConsoleStream::xsputn(const char *p, std::streamsize n)
{
    m_string.append(p, p + n);

    int pos = 0;
    while (pos != std::string::npos)
    {
        pos = m_string.find('\n');
        if (pos != std::string::npos)
        {
            std::string tmp(m_string.begin(), m_string.begin() + pos);
            emit flushStream(QString(tmp.c_str()).append(line_break));
            m_string.erase(m_string.begin(), m_string.begin() + pos + 1);
        }
    }

    return n;
}