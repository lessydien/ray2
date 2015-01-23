#ifndef CONSOLESTREAM_H
#define CONSOLESTREAM_H

#include <iostream>
#include <qobject.h>

class ConsoleStream : public QObject, public std::basic_streambuf<char>
{
	Q_OBJECT
public:
	ConsoleStream(std::ostream &stream, QString lineBreak = "\n");
	~ConsoleStream(void);

signals:
	void flushStream(QString);

protected:

    //! this method overwrites a corresponding method in basic_streambuf class and is invoked, if buffer risks to overflow
    virtual int_type overflow(int_type v)
    {
        if (v == '\n')
        {
            emit flushStream(QString(m_string.c_str()));
            m_string.erase(m_string.begin(), m_string.end());
        }
        else
        {
            m_string += v;
        }

        return v;
    };

    virtual std::streamsize xsputn(const char *p, std::streamsize n);

private:
    std::ostream &m_stream;     /*!<  standard-ostream which is observed by this instance */
    std::streambuf *m_old_buf;  /*!<  content of stream at time when this instance starts the observation of the stream is stored here and re-given to the stream, when this instance is destroyed */
    std::string m_string;       /*!<  buffer string, containing parts of the stream which have not been emitted yet */
//    tMsgType msg_type;          /*!<  message type of enumeration tMsgType which belongs to this instance of QDebugStream */
    QString line_break;         /*!<  string representation of a line break (default: \n) */

};


#endif