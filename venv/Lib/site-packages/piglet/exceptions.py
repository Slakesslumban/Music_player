# Copyright 2016 Oliver Cope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PigletError(Exception):
    """
    Base exception for all piglet templating errors
    """

    lineno = None
    filename = None

    def __str__(self):
        if self.lineno or self.filename:
            return "{} in {}, line {}".format(self.args[0], self.filename, self.lineno)
        return super(PigletError, self).__str__()

    def set_location(self, filename=None, lineno=None):
        if filename is not None and self.filename == self.__class__.filename:
            self.filename = filename
        if lineno is not None and self.lineno == self.__class__.lineno:
            self.lineno = lineno


class PigletParseError(PigletError):
    """
    An error occurred while parsing the input HTML.
    """

    lineno = "(unknown line number)"
    filename = "(unknown filename)"

    def __init__(self, message, filename=None, lineno=None):
        super(PigletParseError, self).__init__(message)
        self.set_location(filename, lineno)


class TemplateNotFound(PigletError):
    """
    Template could not found in specified location
    """


class UndefinedError(PigletError):
    """
    The requested variable is not defined in this context
    """
