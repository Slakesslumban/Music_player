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

from ast import (
    AST,
    Attribute,
    For,
    FunctionDef,
    Load,
    Module,
    Name,
    Param,
    Store,
    With,
    keyword,
    parse,
)
from functools import reduce
from textwrap import dedent
import sys

try:
    from ast import Try as _Try

    _TryExcept = None
except ImportError:
    from ast import TryExcept as _TryExcept

    _Try = None

try:
    from ast import arg
except ImportError:
    arg = None

StoreName = lambda n: Name(id=n, ctx=Store())
LoadName = lambda n: Name(id=n, ctx=Load())


def LoadAttribute(symbol, attr=None):

    if attr is None:
        symbols = symbol.split(".")
        return reduce(LoadAttribute, symbols)

    if isinstance(symbol, AST):
        return Attribute(value=symbol, attr=attr, ctx=Load())
    return Attribute(value=LoadName(symbol), attr=attr, ctx=Load())


def add_arg_default(fn, arg, default):
    fn.args.args.append(arg)
    fn.args.defaults.append(default)


def TryExcept(body, handlers, orelse=None):
    if orelse is None:
        orelse = []
    if _Try:
        return _Try(body=body, handlers=handlers, orelse=orelse, finalbody=[])
    else:
        return _TryExcept(body=body, handlers=handlers, orelse=orelse)


def add_kwarg(call, value):
    """
    Add a **kwarg parameter to a function call node.

    This has different syntax in python3.5+
    """
    if not isinstance(value, AST):
        value = LoadName(value)

    if sys.version_info > (3, 5,):
        call.keywords.append(keyword(arg=None, value=value))
    else:
        call.kwargs = value


def parse_and_strip(pysrc):
    """
    :param pysrc: arbitrary python source code
    :returns: a list of AST nodes, with location information stripped
    """
    pysrc = dedent(pysrc)
    return strip_locations(parse(pysrc)).body


def make_arg(name):
    """
    Shim for differing Python 2 + 3 representations of function args
    """
    if arg is not None:
        return arg(arg=name, annotation=None)
    return Name(id=name, ctx=Param())


def make_kwarg(name):
    """
    Shim for differing Python 2 + 3 representations of function args
    """
    if arg is not None:
        return arg(arg=name, annotation=None)
    return name


def strip_locations(astnode):
    """
    Strip lineno/col_offset annotations from ast.

    This is so that sections of ast generated via parse (which
    has location info embedded) don't mess up the line numbering
    added during compilation
    """
    for n, _ in astwalk(astnode):
        if hasattr(n, "lineno"):
            del n.lineno
        if hasattr(n, "col_offset"):
            del n.col_offset
    return astnode


def get_comparison(pysrc):
    """
    :param pysrc: source code of a comparison, eg 'a == b'
    :returns: an AST node suitable for passing to the constructor of ``If``
    """
    return parse_and_strip("if {}: pass".format(pysrc))[0].test


# List of nodes that indicate a new lexical scope
CREATES_SCOPE = {Module, FunctionDef, For, With}

# Walk fields in this priority order:
# - value/targets: emit the value of an assignment before its target names
# - args/body: emit function args before the body
walk_field_priority = ["value", "targets", "args", "body"]
walk_sort_order = {n: priority for priority, n in enumerate(walk_field_priority)}

# Fields containing nodes to be considered children of the current node.
walk_child_attrs = {"body", "args"}


def astwalk(node, ancestors=None, exclude=lambda ancestors, node, attr: False):
    """
    Walk the AST, generating tuples of ``(node, ancestors)``.

    AST nodes are walked in an order conforming to python's scoping rules:
    nodes are yielded only after all the nodes in their scope have been
    yielded.

    :param exclude: a callable that will be passed each combination of
                    (ancestors, node, attribute_name). If the callable
                    returns True, the given attribute will be skipped in the
                    traversal.

                    For example to exclude for loop contents you would pass::

                        exclude=lambda a, n, attr: \
                            (n.__class__ == For and attr == 'body')
    """

    def get_astnodes_in_order(
        node, child_attrs=walk_child_attrs, sort_order=walk_sort_order
    ):
        """
        Generates tuples of ``(astnode, is_child)``

        is_child signals whether the astnode is within the scope of ``node``.
        For example, consider::

            for x in y():
                print(x)

        The print statement is lexically scoped with in the for loop, and
        has access to the symbol ``x``. The call to ``y()`` is still within the
        parent's scope and doesn't have access to ``x``.

        Therefore this function would generate tuples along the lines of this
        pseudo-AST:

            (Call(func='y'), False)
            (Name(id='x', ctx=Store()), False)
            (For(...), False)
            (Call(func='print', args='x'), True)

        """
        attrs = set(node._fields) - child_attrs
        attrs = sorted(attrs, key=lambda a: sort_order.get(a, len(sort_order)))
        child_attrs = sorted(
            child_attrs, key=lambda a: sort_order.get(a, len(sort_order))
        )
        for item in attrs:
            item = getattr(node, item, [])

            if isinstance(item, list):
                for i in item:
                    yield i, False

            elif isinstance(item, AST):
                yield item, False

        yield node, False

        for item in child_attrs:
            item = getattr(node, item, [])

            if isinstance(item, list):
                for i in item:
                    yield i, True

            elif isinstance(item, AST):
                yield item, True

    ancestors = ancestors or tuple()
    scoped = node.__class__ in CREATES_SCOPE
    left = []
    right = []
    for n, is_child in get_astnodes_in_order(node):
        if is_child:
            right.append(n)
        else:
            left.append(n)

    for item in left:
        if item is node:
            yield item, ancestors
        else:
            for sub in astwalk(item, ancestors):
                yield sub

    if scoped:
        ancestors = ancestors + (node,)

    for item in right:
        for sub in astwalk(item, ancestors):
            yield sub


def add_locations(node):
    l, c = 1, 1
    for n, ancestors in astwalk(node):
        if hasattr(n, "lineno"):
            l = n.lineno
        else:
            n.lineno = l
        if hasattr(n, "column"):
            c = n.col_offset
        else:
            n.col_offset = c
    return node
