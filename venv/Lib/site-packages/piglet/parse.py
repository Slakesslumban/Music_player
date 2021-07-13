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

from pkg_resources import resource_string
from collections import OrderedDict

import attr
import parsley

from piglet.exceptions import PigletParseError
from piglet.position import Position


@attr.s
class ParseItem(object):
    pos = attr.ib(default=None)

    def set_pos(self, pos):
        self.pos = pos
        return pos.advance(self.source)

    @property
    def source(self):
        raise NotImplementedError()


@attr.s
class OpenTag(ParseItem):

    qname = attr.ib(default=None)

    #: Whitespace between the name and the first attribute
    space = attr.ib(default=None)
    attrs = attr.ib(default=attr.Factory(OrderedDict), converter=OrderedDict)

    end = attr.ib(default=">")

    def set_pos(self, pos):
        cursor = pos.advance("<{}{}".format(self.qname, self.space))
        for item in self.attrs.values():
            cursor = item.set_pos(cursor)
        return super(OpenTag, self).set_pos(pos)

    @property
    def source(self):
        attrs = "".join(v.source for v in self.attrs.values())
        return "<{}{}{}{}".format(self.qname, self.space, attrs, self.end)


@attr.s
class Attribute(ParseItem):
    name = attr.ib(default=None)
    value = attr.ib(default=None)
    quote = attr.ib(default=None)
    space1 = attr.ib(default=None)
    space2 = attr.ib(default=None)
    space3 = attr.ib(default=None)

    #: The start position of the attribute's value
    value_pos = attr.ib(default=None)

    def __init__(self, *args, **kwargs):
        super(Attribute, self).__init__(*args, **kwargs)

    def set_pos(self, pos):
        self.value_pos = pos.advance(
            "{0.name}{0.space1}={0.space2}{0.quote}".format(self)
        )
        return super(Attribute, self).set_pos(pos)

    @property
    def source(self):
        return (
            "{0.name}{0.space1}="
            "{0.space2}{0.quote}{0.value}{0.quote}{0.space3}".format(self)
        )


@attr.s
class CloseTag(ParseItem):
    qname = attr.ib(default=None)

    @property
    def source(self):
        return "</{}>".format(self.qname)


class OpenCloseTag(OpenTag):
    @property
    def source(self):
        attrs = "".join(v.source for v in self.attrs.values())
        return "<{}{}{}>".format(self.qname, self.space, attrs)


@attr.s
class Comment(ParseItem):
    content = attr.ib(default=None)

    @property
    def source(self):
        return "<!--{}-->".format(self.content)


@attr.s
class Text(ParseItem):
    content = attr.ib(default=None)
    cdata = attr.ib(default=False)

    @property
    def source(self):
        return self.content


@attr.s
class Entity(ParseItem):
    reference = attr.ib(default=None)

    @property
    def source(self):
        return self.reference


@attr.s
class PI(ParseItem):
    target = attr.ib(default=None)
    content = attr.ib(default=None)

    @property
    def source(self):
        return "<?{}{}?>".format(self.target, self.content)


@attr.s
class Declaration(ParseItem):
    content = attr.ib(default=None)

    @property
    def source(self):
        return "<!{}>".format(self.content)


@attr.s
class CDATA(ParseItem):
    content = attr.ib(default=None)

    @property
    def source(self):
        return "<![CDATA[{}]]>".format(self.content)


parser_ns = locals()
html_grammar = resource_string(__name__, "grammar.txt").decode("UTF-8")
html_parser = parsley.makeGrammar(html_grammar, parser_ns)


def parse_html(source):
    try:
        return add_positions(list(flatten(html_parser(source).html())))
    except parsley.ParseError as e:
        raise PigletParseError(e.formatError())


def flatten(parse_result):
    for item in parse_result:
        if isinstance(item, list):
            for i2 in flatten(item):
                yield i2
        else:
            yield item


def add_positions(parse_result):
    """
    Add position annotations to each parsed item
    """
    cursor = Position(1, 1)
    for item in parse_result:
        cursor = item.set_pos(cursor)
    return parse_result
