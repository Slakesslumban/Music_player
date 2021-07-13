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

from itertools import chain
from pkg_resources import resource_string
import re

import attr
import parsley


#: Grammar to parse semicolon separated values in WithNodes
ssv_grammar = resource_string(__name__, "grammar_ssv.txt").decode("UTF-8")
ssv_parser = parsley.makeGrammar(ssv_grammar, {})


@attr.s
class IntermediateNode(object):
    """
    An intermediate representation of the template structure.

    A template such as::

        <py:for each="i, x in enumerate(xs)">
            <a py:if="x.href" href="$x.href">
                link to ${x.name}
            </a>
        </py:for>

    Could be modelled as::

        ForNode(
            each="i, x in enumerate(xs)",
            children=[
                IfNode(expr='x.href',
                       children=[TextNode('<a href="'),
                                 InterpolateNode('x.href'),
                                 TextNode('">link to "),
                                 InterpolateNode('x.name'),
                                 TextNode('</a>')])])
    """

    pos = attr.ib(default=None)
    children = []

    @classmethod
    def factory(cls, *args, **kwargs):
        raise NotImplementedError(cls)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        return self

    def find(self, cls=object, test=lambda node: True):
        """
        Walk this node and all children, yielding matching nodes

        :param cls: Match only nodes of this class
        :param test: Match only nodes for which this predicate returns True
        """
        for parent, node, idx in walk_tree_pre(self):
            if isinstance(node, cls) and test(node):
                yield node

    @property
    def all_children(self):
        """
        Some nodes may have multiple lists of children (eg an IfNode
        has both ``children`` and ``else_``. The ``all_children``
        property should return an iterable over all child nodes for the
        purpose of tree walking
        """
        return self.children


@attr.s
class ContainerNode(IntermediateNode):
    """
    A generic container node
    """

    children = attr.ib(attr.Factory(list))

    @classmethod
    def factory(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def append(self, child):
        """
        Add child to this node's list of children.
        """
        self.children.append(child)

    def extend(self, children):
        self.children.extend(children)

    def tip(self):
        """
        Find the newest tip of the tree
        """
        if not self.children:
            return self

        if isinstance(self.children[-1], ContainerNode):
            return self.children[-1].tip()

        return self.children[-1]


class DirectiveNode(object):
    @classmethod
    def factory(cls, **kwargs):
        return cls(**kwargs)


class RootNode(ContainerNode):
    "Container node for an entire document"

    @classmethod
    def factory(cls):
        return cls()


@attr.s(init=False)
class TextNode(IntermediateNode):

    content = attr.ib(default=None)

    def __init__(self, content=None, pos=None):
        self.content = content
        self.pos = pos

    @classmethod
    def factory(cls, content=None, pos=None):
        return cls(content, pos)


@attr.s
class BlockNode(ContainerNode):
    """
    Models a py:block directive
    """

    name = attr.ib(default=None)

    @classmethod
    def factory(cls, name):
        return cls(name=name)


@attr.s
class IfNode(DirectiveNode, ContainerNode):
    """
    Models a py:if ... py:else directive
    """

    test = attr.ib(default=None)
    else_ = attr.ib(default=None)

    @property
    def all_children(self):
        if self.else_:
            return chain(self.children, [self.else_])
        return self.children


class ElseNode(DirectiveNode, ContainerNode):
    """
    The else part of a ``py:if ... py:else`` directive.
    """


@attr.s
class DefNode(DirectiveNode, ContainerNode):
    """
    Models a py:def directive
    """

    function = attr.ib(default=None)


@attr.s
class ExtendsNode(ContainerNode):
    """
    Models a py:extends directive
    """

    href = attr.ib(default=None)

    @classmethod
    def factory(cls, href):
        return cls(href=href)


@attr.s
class IncludeNode(DirectiveNode, IntermediateNode):
    """
    Models a py:include directive
    """

    href = attr.ib(default=None)


@attr.s
class ChooseNode(DirectiveNode, ContainerNode):
    """
    Models a py:choose directive.

    A choose node may have When, Otherwise and Text children.
    If :attribute:`ChooseNode.test` contains an expression, the value of that
    expression is is compared to any values contained in
    :attribute:`WhenNode.test` children. The first matching :class:`When` node
    is rendered and the others are dropped.

    If :attribute:`ChooseNode.test` is empty, each contained
    :attribute:`WhenNode.test` is evaluated as a boolean expression and the
    first truthful result is rendered.

    If no :class:`WhenNode` is rendered, any :class:`OtherwiseNode` directives
    will be rendered.
    """

    test = attr.ib(default=None)


@attr.s
class WhenNode(DirectiveNode, ContainerNode):
    """
    Models a py:when directive, and must be contained with a
    :class:`ChooseNode` directive.
    """

    test = attr.ib(default=None)


class OtherwiseNode(DirectiveNode, ContainerNode):
    pass


@attr.s
class ForNode(DirectiveNode, ContainerNode):
    """
    Models a py:for directive.

    The expression in :attribute:`ForNode.each` must be in the form
    `<target> in <iterator>`, and is used to generate a python for loop.
    """

    each = attr.ib(default=None)


@attr.s
class WithNode(DirectiveNode, ContainerNode):
    """
    Models a py:with directive.

    The expression in :attribute:`WithNode.vars` must be semicolon separated
    list of variable assignments, for example:

        WithNode(vars='x = 1; y = x * 2')

    The assigned variables will be available only within the scope of the
    directive
    """

    vars = attr.ib(default=[])

    def get_pairs(self):
        """
        Return the configured variables as a list of (target, expr) pairs.
        """
        if isinstance(self.vars, list):
            return self.vars

        values = ssv_parser(self.vars).values()
        values = [tuple(s.strip() for s in str(item).split("=", 1)) for item in values]
        return values


@attr.s
class InterpolateNode(DirectiveNode, IntermediateNode):
    """
    Renders a python expression interpolation
    """

    value = attr.ib(default=None)
    autoescape = attr.ib(default=True)


class NullNode(ContainerNode):
    """
    Used to account for text that should not appear in the compiled template,
    but still needs a node to keep the line numbering correct.
    """


@attr.s
class ImportNode(DirectiveNode, ContainerNode):
    href = attr.ib(default=None)
    alias = attr.ib(default=None)


@attr.s
class InlineCodeNode(DirectiveNode, ContainerNode):
    pysrc = attr.ib(default=None)


@attr.s
class FilterNode(DirectiveNode, ContainerNode):
    function = attr.ib(default=None)


@attr.s
class TranslationNode(DirectiveNode, ContainerNode):
    message = attr.ib(default=None)
    comment = attr.ib(default=None)

    def get_msgstr(self):

        if self.message:
            return self.message

        s = []
        for name, item in self.named_children():
            if name is None:
                s.append(item.content)
            else:
                s.append("${{{}}}".format(name))
        s = "".join(s)
        return re.sub("[ \t\r\n]+", " ", s).strip()

    def named_children(self):
        """
        Return a tuples of ('placeholder_name', imnode).
        For TextNode children, placeholder_name will be None.
        """
        dyn_index = 1
        for item in self.children:
            if isinstance(item, TextNode):
                yield (None, item)
            elif isinstance(item, TranslationPlaceholder):
                yield (item.name, item)
            elif isinstance(item, InterpolateNode):
                yield (item.value.strip(), item)
            else:
                yield ("dynamic.{}".format(dyn_index), item)
                dyn_index += 1


@attr.s
class TranslationPlaceholder(DirectiveNode, ContainerNode):
    name = attr.ib(default=None)


@attr.s
class Call(DirectiveNode, ContainerNode):
    """
    A python function call
    """

    function = attr.ib(default=None)


@attr.s
class CallKeyword(DirectiveNode, ContainerNode):
    """
    A keyword argument to a function :class:`Call`
    that may contain an arbitrary template snippet
    """

    name = attr.ib(default=None)


@attr.s
class Comment(DirectiveNode, ContainerNode):
    """
    A comment block that will be removed from the template output
    """


def _walk_tree(order, n, parent, pos):
    assert order in {"pre", "post"}
    if order == "pre":
        yield parent, n, pos
        children = enumerate(n.all_children)
    else:
        children = reversed(list(enumerate(n.all_children)))

    for index, sub in children:
        for item in _walk_tree(order, sub, n, index):
            yield item
    if order == "post":
        yield parent, n, pos


def walk_tree_post(n, parent=None, pos=0):
    """
    Walk the intermediate tree in a post order traversal.

    Yields tuples of (parent, node, index).

    A post order traversal is chosen so that nodes may be deleted
    or merged without affecting the subsequent traversal.
    """
    return _walk_tree("post", n, parent, pos)


def walk_tree_pre(n, parent=None, pos=0):
    """
    Walk the intermediate tree in a pre order traversal.

    Yields tuples of (parent, node, index).
    """
    return _walk_tree("pre", n, parent, pos)


walk_tree = walk_tree_pre
