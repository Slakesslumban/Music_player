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

from functools import partial
from pkg_resources import resource_string
import re

import attr
import parsley

from piglet import parse
from piglet import compilexml as cx
from piglet import intermediate as im
from piglet.exceptions import PigletParseError


@attr.s
class Statement(parse.ParseItem):

    EMPTY_TAGS = "import"

    name = attr.ib(default=None)
    args = attr.ib(default=None)
    space1 = attr.ib(default=None)
    space2 = attr.ib(default=None)
    end_tag = attr.ib(default=None)
    end_tag_pos = attr.ib(default=None)
    children = attr.ib(default=attr.Factory(list))
    ws_before = attr.ib(default=None)
    ws_after = attr.ib(default=None)

    @property
    def source(self):
        return "{{%{0.space1}{0.name}{0.space2}{0.args}%}}".format(self)

    @property
    def raw_args(self):
        return "".join(
            a.source if isinstance(a, QuotedString) else a for a in self.args
        )

    def is_open_tag(self):
        return not self.name.startswith("end")

    def is_openclose_tag(self):
        return self.name in self.EMPTY_TAGS

    def is_close_tag(self):
        return self.name.startswith("end")


@attr.s
class QuotedString(parse.ParseItem):
    quote = attr.ib(default=None)
    content = attr.ib(default=None)

    @property
    def source(self):
        return "{0.quote}{0.content}{0.quote}".format(self)


@attr.s
class Container(object):
    children = attr.ib(default=attr.Factory(list))


parser_ns = {
    "Statement": Statement,
    "Text": parse.Text,
    "QuotedString": QuotedString,
}
tt_grammar = resource_string(__name__, "grammar_tt.txt").decode("UTF-8")
tt_parser = parsley.makeGrammar(tt_grammar, parser_ns)


def parse_tt(source):
    try:
        return parse.add_positions(list(tt_parser(source).template()))
    except parsley.ParseError as e:
        raise PigletParseError(e.formatError())


def make_tree(parse_result):
    """
    From ``parse_result``, a list of parsed tokens create a tree of
    elements
    """
    stack = [Container()]
    head = stack[-1]
    for item in parse_result:
        if isinstance(item, Statement) and item.is_openclose_tag():
            head.children.append(item)

        elif isinstance(item, Statement) and item.is_open_tag():
            head.children.append(item)
            stack.append(item)
            head = item

        elif isinstance(item, Statement) and item.is_close_tag():
            # else statements don't have associated end tags - the end tag
            # should cause the parent if statement to close instead
            if head.name == "else":
                head = stack.pop()
            name = item.name[3:]
            if name and name != head.name:
                raise parse.PigletParseError(
                    "Unexpected {} at {}:{}".format(
                        item.source, item.pos.line, item.pos.char
                    )
                )
            head.end_tag = item
            head.end_tag_pos = item.pos
            stack.pop()
            head = stack[-1]
        else:
            head.children.append(item)
    if len(stack) > 1:
        raise parse.PigletParseError(
            "Missing end tag for {}, "
            "opened at {}:{}".format(
                stack[-1].source, stack[-1].pos.line, stack[-1].pos.char
            )
        )
    return stack[0]


def compile_intermediate(parse_result):
    tree = make_tree(parse_result)
    strip_ws_on_directive_lines(tree)
    compiled = _compile_node(tree)
    cx._strip_unwanted_nodes(compiled)
    cx._concatentate_adjacent_text_nodes(compiled)
    cx._postprocess(compiled)
    return compiled


def strip_ws_on_directive_lines(node):

    # Remove whitespace until after first newline
    trim_line_start = partial(re.compile(r"\A[ \t]*(\r\n|\n|\r)").sub, r"")

    # Remove whitespace after last newline
    trim_line_end = partial(re.compile(r"([\r\n])[ \t]*\Z").sub, r"\1")

    # Remove leading whitespace
    trim_all_start = partial(re.compile(r"\A\s*", re.S).sub, r"")

    # Remove trailing whitespace
    trim_all_end = partial(re.compile(r"\s*\Z", re.S).sub, r"")

    trim_nothing = lambda s: s  # noqa

    trim_start = {"+": trim_nothing, "-": trim_all_start, None: trim_line_start}
    trim_end = {"+": trim_nothing, "-": trim_all_end, None: trim_line_end}

    if isinstance(node, Statement):
        if node.children:
            # Remove whitespace until after first newline inside a statement
            c = node.children[0]
            if isinstance(c, parse.Text):
                c.content = trim_start[node.ws_after](c.content)

            # Remove whitespace after the last newline inside a statement
            c = node.children[-1]
            if isinstance(c, parse.Text):
                c.content = trim_end[node.end_tag.ws_before](c.content)

    for p, n in zip(node.children, node.children[1:]):

        # Remove whitespace preceding a statement
        if isinstance(p, parse.Text) and isinstance(n, Statement):
            p.content = trim_end[n.ws_before](p.content)

        # Remove whitespace following a statement
        if isinstance(p, Statement) and isinstance(n, parse.Text) and p.end_tag:
            n.content = trim_start[p.end_tag.ws_after](n.content)

    for item in node.children:
        if hasattr(item, "children"):
            strip_ws_on_directive_lines(item)


def compile_extends(stmt, href):
    return im.ExtendsNode(
        href=href, pos=stmt.pos, children=[_compile_node(c) for c in stmt.children]
    )


def compile_block(stmt, name):
    return im.BlockNode(
        name=name, pos=stmt.pos, children=[_compile_node(c) for c in stmt.children]
    )


def compile_for(stmt, *args):
    return im.ForNode(
        each=stmt.raw_args,
        pos=stmt.pos,
        children=[_compile_node(c) for c in stmt.children],
    )


def compile_def(stmt, *args):
    return im.DefNode(
        function=stmt.raw_args,
        pos=stmt.pos,
        children=[_compile_node(c) for c in stmt.children],
    )


def compile_if(stmt, *args):
    n = im.IfNode(test=stmt.raw_args, pos=stmt.pos)

    children = stmt.children
    if children:
        last = children[-1]
        if isinstance(last, Statement) and last.name == "else":
            children = children[:-1]
            n.else_ = im.ElseNode(
                pos=last.pos, children=[_compile_node(c) for c in last.children]
            )
    n.children = [_compile_node(c) for c in children]
    return n


def compile_with(stmt, *args):
    return im.WithNode(
        vars=stmt.raw_args,
        children=[_compile_node(c) for c in stmt.children],
        pos=stmt.pos,
    )


def compile_trans(stmt, message=None):
    return im.TranslationNode(
        message=message,
        children=[_compile_node(c) for c in stmt.children],
        pos=stmt.pos,
    )


def compile_transname(stmt, name):
    return im.TranslationPlaceholder(
        name=name, children=[_compile_node(c) for c in stmt.children], pos=stmt.pos
    )


def compile_import(stmt, href, as_, alias, *args):
    if as_ != "as" or args:
        raise PigletParseError(
            "Syntax error: "
            "should be \"import 'path/to/template.txt' as alias_name\"."
        )

    return im.ImportNode.factory(href=href, alias=alias, pos=stmt.pos)


DIRECTIVES = {
    "extends": compile_extends,
    "block": compile_block,
    "for": compile_for,
    "if": compile_if,
    "def": compile_def,
    "with": compile_with,
    "trans": compile_trans,
    "transname": compile_transname,
    "import": compile_import,
}


def split_args(args):
    for a in args:
        if isinstance(a, QuotedString):
            yield a.content
        else:
            for b in a.strip().split():
                if b:
                    yield b


def _compile_node(node):
    """
    Compile a single node to intermediate representation
    """

    if isinstance(node, Container):
        root = im.ContainerNode()
        root.extend(map(_compile_node, node.children))
        return root

    elif isinstance(node, parse.Text):
        return cx._compile_text(node)

    elif isinstance(node, Statement):
        args = split_args(node.args)
        return DIRECTIVES[node.name](node, *args)

    else:
        raise NotImplementedError(node)
