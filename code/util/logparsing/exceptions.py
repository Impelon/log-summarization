class LogParsingException(Exception):
    """
    Base class for exceptions raised by the logparsing module.
    """

class LogLineException(LogParsingException):
    """
    Base class for exceptions caused by a log line.

    Attributes:
        line: A string representing the problematic line.
        line_number: The number of line in the supplied log.
    """

    def __init__(self, line, line_number):
        self.line = line
        self.line_number = line_number
        super(LogLineException, self).__init__(self.construct_message())

    def construct_message(self):
        return "'{}' at position {}".format(self.line.strip(), self.line_number)

    def __reduce__(self):
        return type(self), (self.line, self.line_number)


class ImproperlyFormattedLogLineError(LogLineException):
    """
    Exception that is raised when a log line does not adhere to a specific format.
    """
