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

from copy import copy
from pkg_resources import resource_string

import attr
import parsley

from piglet.exceptions import PigletParseError


@attr.s
class Interpolation(object):
    source = attr.ib()
    value = attr.ib()

    autoescape = True

    def noescape(self):
        ob = copy(self)
        ob.autoescape = False
        return ob


parser_ns = locals()
interp_grammar = resource_string(__name__, "grammar_interpolate.txt").decode("UTF-8")
interp_parser = parsley.makeGrammar(interp_grammar, parser_ns)


def parse_interpolations(source):
    try:
        return interp_parser(source).text_with_interpolations()
    except parsley.ParseError as e:
        raise PigletParseError(e.formatError())
