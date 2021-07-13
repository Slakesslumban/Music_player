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

"""
Compile XML to an intermediate parse tree
"""
from copy import copy
from itertools import chain
from itertools import product
from itertools import takewhile
import textwrap
import re

import attr

from piglet import intermediate as im
from piglet import interpolate
from piglet import parse
from piglet.exceptions import PigletParseError

XMLNS = {"py": "py", "i18n": "i18n"}

# Patterns to match whitespace (but not non-breaking spaces or other unicode
# whitespace)
ENTIRELY_WS = re.compile(r"^[ \t\r\n]*$")
SURROUNDING_WS = re.compile(r"^" r"([ \t\r\n]*)" r"(.*[^ \t\r\n])" r"([ \t\r\n]*)$")
WS_L = re.compile(r"^([ \t\r\n]*)(.*[^ \t\r\n].*)?$", re.S)
WS_R = re.compile(r"(^|(?:.*[^ \t\r\n]))([ \t\r\n]*)$")
CRNL = re.compile(r"[\r\n]")

#: List of attributes that should be omitted if the interpolated value
#: evaluates to None or False.
#: For example '<input checked="$checked"/>' should be rendered as '<input/>'
#: in the case that checked == False.
HTML_EMPTY_ATTRS = {
    "checked",
    "declare",
    "defer",
    "disabled",
    "ismap",
    "multiple",
    "nohref",
    "readonly",
    "selected",
}

try:
    from html import unescape
except ImportError:
    import htmlentitydefs

    def unescape(text):
        """
        Removes HTML or XML character references and entities from a text string.

        @param text The HTML (or XML) source text.
        @return The plain text, as a Unicode string, if necessary.

        Adapted from http://effbot.org/zone/re-sub.htm#unescape-html
        """

        def fixup(m):
            text = m.group(0)
            if text[:2] == "&#":
                try:
                    if text[:3] == "&#x":
                        return unichr(int(text[3:-1], 16))  # noqa
                    else:
                        return unichr(int(text[2:-1]))  # noqa
                except ValueError:
                    pass
            else:
                try:
                    text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])  # noqa
                except KeyError:
                    pass
            return text

        return re.sub(r"&#?\w+;", fixup, text)


ustr = type(u"")

WHITESPACE = "py:whitespace"


class ElsePlaceholder(im.DirectiveNode, im.ContainerNode):
    """
    Placeholder for the else part of a <py:if> block
    """


@attr.s
class Directive(object):

    REQUIRED = object()

    #: Tuple of (tag-name, attribute-name). Either can be blank if the
    #: directive may be used only as a tag or attribute.
    name = attr.ib()

    #: Attribute name for required data when used as a tag.
    #: Eg for 'py:if', this would be 'test'. If None, the tag name is
    #: used
    attr_ = attr.ib()

    factory = attr.ib()

    #: If no data attribute is present, what should be the default?
    default = attr.ib(default=REQUIRED)

    #: If true, any data attribute should be discarded
    empty = attr.ib(default=False)

    #: List of (attribute name, required) tuples for additional data
    #: Eg for 'py:import', this would be [('alias', True)]
    extra_attrs = attr.ib(default=attr.Factory(list))

    inner = attr.ib(default=False)
    ns = attr.ib(default=XMLNS["py"])
    aliases = attr.ib(default=tuple())

    attr = attr_
    del attr_

    @property
    def qname(self):
        return "{}:{}".format(self.ns, self.name)

    def get_extra_attrs(self, data, el):
        """
        Return a dict of additional attributes to process
        """
        return {}

    def get_strip_condition(self, data, el):
        """
        Return python src of a condition to strip the tag
        """
        return None

    def get_tagname_expr(self, data, el):
        """
        Return a python expression for the tag name, or None
        """
        return None

    def q(self, n):
        """
        Qualify name ``n`` by putting it in the directive's configured
        namespace
        """
        return "{}:{}".format(self.ns, n)

    def parse(self, el):
        """
        Return a tuple of ``(data, remaining)`` for the given element.
        If element does not match this directive, return None.
        """
        for name in chain([self.name], self.aliases):
            if isinstance(name, tuple):
                tagname, attrname = name
            else:
                tagname = attrname = name

            if tagname:
                tagname = self.q(tagname)
            if attrname:
                attrname = self.q(attrname)

            parsed = self._parse(tagname, attrname, el)
            if parsed is not None:
                return parsed

        return None

    def _parse(self, tagname, attrname, el):

        data = {}
        pos = el.pos
        if el.qname == tagname:
            remaining = Fragment(
                children=el.children,
                pos=el.pos,
                end=el.end,
                close_tag_pos=el.close_tag_pos,
                attrs=el.attrs,
            )
            remaining.close_tag = el.close_tag
            if not self.empty:
                try:
                    data = {self.attr: remaining.attrs.pop(self.q(self.attr)).value}
                except KeyError:
                    if self.default is self.REQUIRED:
                        raise PigletParseError(
                            "Missing attribute {!r} in element {}".format(
                                self.q(self.attr), el
                            ),
                            lineno=el.pos.line,
                        )
                    else:
                        data = {self.attr: self.default}

        elif attrname in el.attrs:
            pos = el.attrs[attrname].value_pos
            remaining = copy(el)
            remaining.attrs = copy(el.attrs)
            if self.empty:
                remaining.attrs.pop(attrname)
            else:
                data = {self.attr: remaining.attrs.pop(attrname).value}

        else:
            return None

        for extra, required in self.extra_attrs:
            try:
                data[extra] = remaining.attrs.pop(self.q(extra)).value
            except KeyError:
                if required:
                    raise PigletParseError(
                        "Missing attribute {!r} in element {}".format(
                            self.q(extra), el
                        ),
                        lineno=el.pos.line,
                    )

        data = {k: unescape(v) for k, v in data.items()}
        return (data, remaining, pos)


class PyAttrsDirective(Directive):
    def get_extra_attrs(self, data, element):
        return data["attrs"]


class PyStripDirective(Directive):
    def get_strip_condition(self, data, el):
        if data["strip"].strip() == "":
            return "True"
        else:
            return data["strip"]


class PyTagDirective(Directive):
    def get_tagname_expr(self, data, el):
        return data["tag"]


def make_i18n_substitution(name, expr=None):
    """
    Turn a <i18n:s name="foo" expr="foo"/> into intermediate nodes
    """
    if expr and not name:
        name = expr
    n = im.TranslationPlaceholder(name=name)
    if expr:
        n.append(im.InterpolateNode(value=expr))
    return n


def make_i18n_from_comment(comment):
    return im.TranslationNode(comment=comment)


directives = [
    Directive(name="comment", attr="comment", empty=True, factory=im.Comment.factory),
    Directive(name="with", attr="vars", factory=im.WithNode.factory),
    Directive(name="def", attr="function", factory=im.DefNode.factory),
    Directive(name="extends", attr="href", factory=im.ExtendsNode.factory),
    Directive(name="include", attr="href", factory=im.IncludeNode.factory),
    Directive(name="block", attr="name", inner=True, factory=im.BlockNode.factory),
    Directive(
        name="import",
        attr="href",
        extra_attrs=[("alias", True)],
        factory=im.ImportNode.factory,
    ),
    Directive(name="when", attr="test", factory=im.WhenNode.factory),
    Directive(
        name="case", attr="value", factory=lambda value: im.WhenNode.factory(test=value)
    ),
    Directive(
        name="otherwise", attr="otherwise", empty=True, factory=im.OtherwiseNode.factory
    ),
    Directive(name="else", attr="else", empty=True, factory=ElsePlaceholder),
    Directive(name="for", attr="each", factory=im.ForNode.factory),
    Directive(name="if", attr="test", factory=im.IfNode.factory),
    Directive(
        name="switch", attr="test", default="True", factory=im.ChooseNode.factory
    ),
    Directive(
        name="choose", attr="test", default="True", factory=im.ChooseNode.factory
    ),
    Directive(name="call", attr="function", inner=True, factory=im.Call.factory),
    Directive(name="keyword", attr="name", factory=im.CallKeyword.factory),
    Directive(
        name="content", attr="value", inner=True, factory=im.InterpolateNode.factory
    ),
    Directive(name="replace", attr="value", factory=im.InterpolateNode.factory),
    Directive(name="filter", attr="function", inner=True, factory=im.FilterNode),
    PyAttrsDirective(name=(None, "attrs"), attr="attrs", factory=None),
    PyStripDirective(name=(None, "strip"), attr="strip", factory=None),
    PyTagDirective(name="tag", attr="tag", factory=None),
    Directive(
        ns=XMLNS["i18n"],
        inner=True,
        name="translate",
        aliases=["trans", "message"],
        attr="message",
        default="",
        extra_attrs=[("comment", False)],
        factory=im.TranslationNode,
    ),
    Directive(
        ns=XMLNS["i18n"],
        name="name",
        attr="name",
        default=None,
        aliases=["s"],
        extra_attrs=[("expr", False)],
        factory=make_i18n_substitution,
    ),
    Directive(
        ns=XMLNS["i18n"],
        inner=True,
        name="comment",
        attr="comment",
        factory=make_i18n_from_comment,
    ),
]


class Element(object):
    def __init__(
        self,
        qname="",
        space="",
        attrs="",
        pos=None,
        end="",
        close_tag_pos=None,
        children=None,
    ):
        self.qname = qname
        self.space = space
        self.attrs = attrs
        self.pos = pos
        self.end = end
        self.close_tag_pos = close_tag_pos
        self.close_tag = None
        self.children = children if children is not None else []
        attrs = self.formatted_attrs()
        if pos is not None:
            self.end_pos = self.pos.advance(
                "<{}{}{}".format(self.qname, self.space, attrs)
            )
        if ":" in qname:
            self.ns = qname.split(":")[0]
        else:
            self.ns = "html"

        for name in list(self.attrs):
            savedname = name
            if ":" not in name:
                name = "{}:{}".format(self.ns, name)
            self.attrs[name] = self.attrs.pop(savedname)

    def __str__(self):
        if self.close_tag:
            return self.open_tag() + "..." + self.close_tag
        else:
            return self.open_tag()

    def __repr__(self):
        return "Element('{}')".format(self)

    def __getitem__(self, k):
        return self.attrs[k]

    def __iter__(self):
        return iter(self.children)

    def append(self, node):
        self.children.append(node)

    def is_dynamic(self):
        namespaces = ["{}:".format(ns) for ns in XMLNS]
        names = chain([self.qname], self.attrs)
        return any(x.startswith(ns) for x, ns in product(names, namespaces))

    def open_tag(self):
        attrs = self.formatted_attrs()
        return "<{}{}{}{}".format(self.qname, self.space, attrs, self.end)

    def formatted_attrs(self):
        if self.attrs:
            return "".join(v.source for v in self.attrs.values())
        return ""


class Fragment(Element):
    def __init__(self, *args, **kwargs):
        if "qname" not in kwargs:
            kwargs["qname"] = "#fragment"
        super(Fragment, self).__init__(*args, **kwargs)

    def open_tag(self):
        return ""


def get_directives(element):
    """
    Analyse a :class:`~piglet.parse.Element`, to yield all associated
    directives tagged on the node.

    Return: a tuple of
            ``([(Directive, associated-data, Position), ...], remaining)``
            where ``remaining`` is an Element object with all directive
            attributes stripped.
    """
    result = []
    remaining = element
    for d in directives:
        p = d.parse(remaining)
        if p is not None:
            data, remaining, pos = p
            result.append((d, data, pos))

    if not remaining.attrs:
        remaining.space = ""

    for name in chain([remaining.qname], remaining.attrs):
        if name.startswith("py:"):
            raise PigletParseError(
                "Unrecognized directive {!r} in element {}".format(name, element),
                lineno=element.pos.line,
            )
    return result, remaining


def make_dom(parse_result):
    """
    From ``parse_result``, a list of parsed tokens, we create a tree of
    elements
    """
    stack = [Fragment()]
    head = stack[-1]
    for item in parse_result:
        if isinstance(item, parse.OpenCloseTag):
            head.append(
                Element(
                    qname=item.qname,
                    space=item.space,
                    attrs=item.attrs,
                    pos=item.pos,
                    end="/>",
                    close_tag_pos=item.pos,
                )
            )

        elif isinstance(item, parse.OpenTag):
            newhead = Element(
                qname=item.qname,
                space=item.space,
                attrs=item.attrs,
                pos=item.pos,
                end=">",
            )
            head.append(newhead)
            stack.append(newhead)
            head = newhead

        elif isinstance(item, parse.CloseTag):
            if item.qname != head.qname:
                raise parse.PigletParseError(
                    "Unexpected {} at {}:{}".format(
                        item.source, item.pos.line, item.pos.char
                    )
                )
            head.close_tag = item.source
            head.close_tag_pos = item.pos
            stack.pop()
            head = stack[-1]

        else:
            head.append(item)
    if len(stack) > 1:
        raise parse.PigletParseError(
            "Missing closing tag for <{}>, "
            "opened at {}:{}".format(
                stack[-1].qname, stack[-1].pos.line, stack[-1].pos.char
            )
        )
    return stack[0]


def strip_spaces_between_directives(
    node,
    element_cls=Element,
    is_directive=(lambda n: isinstance(n, Element) and n.qname.startswith("py:")),
):
    st = None
    ws = []
    FOUND_DIRECTIVE = ["FOUND"]

    for n in node.children:
        is_ws = isinstance(n, parse.Text) and ENTIRELY_WS.match(n.content)
        if st is None:
            if is_directive(n):
                st = FOUND_DIRECTIVE
            else:
                st = None

        elif st == FOUND_DIRECTIVE:
            if is_directive(n):
                continue
            elif is_ws:
                ws.append(n)
            else:
                st = None

        if isinstance(n, element_cls):
            n = strip_spaces_between_directives(n)
    node.children = [n for n in node.children if n not in ws]
    return node


def strip_whitespace(node, strip):

    stripvalues = {"strip": True, "preserve": False}

    def is_whitespace(n):
        return isinstance(n, parse.Text) and ENTIRELY_WS.match(n.content)

    if isinstance(node, Element):
        if node.qname == WHITESPACE:
            strip = stripvalues[node.attrs["py:value"].value]
            node.__class__ = Fragment

        elif WHITESPACE in node.attrs:
            strip = stripvalues[node.attrs[WHITESPACE].value]
            del node.attrs[WHITESPACE]
            if len(node.attrs) == 0:
                node.space = ""

    if strip:
        newchildren = []
        for ix, n in enumerate(node.children):
            if ix > 0:
                preceding = node.children[ix - 1]
            else:
                preceding = None

            if ix < len(node.children) - 1:
                following = node.children[ix + 1]
            else:
                following = None

            if not isinstance(n, parse.Text):
                newchildren.append(n)
                continue

            if preceding is None:
                mo = WS_L.match(n.content)
                if mo:
                    n.content = mo.group(2) or ""
                    n.pos = n.pos.advance(mo.group(1))

            if following is None:
                mo = WS_R.match(n.content)
                if mo:
                    n.content = mo.group(1) or ""

            if isinstance(following, Element):
                mo = WS_R.match(n.content)
                if mo and CRNL.search(mo.group(2)):
                    n.content = mo.group(1)

            if isinstance(preceding, Element):
                mo = WS_L.match(n.content)
                if mo and CRNL.search(mo.group(1)):
                    n.content = mo.group(2)
                    n.pos = n.pos.advance(mo.group(1))

            if n.content:
                newchildren.append(n)
        node.children = newchildren

    for n in node.children:
        if isinstance(n, Element):
            strip_whitespace(n, strip)

    return node


def combine_entities(node):
    """
    Roll entity references up into sibling text nodes.
    This is to allow interpolation parsing to act on the whole
    string across text/entity node boundaries
    """
    acc = None
    newchildren = []

    for n in node.children:
        if hasattr(n, "children"):
            combine_entities(n)
        if isinstance(n, parse.Text):
            if acc is None:
                acc = n
            else:
                acc.content += n.content
        elif isinstance(n, parse.Entity):
            if acc is None:
                acc = parse.Text(pos=n.pos, content=n.source)
            else:
                acc.content += n.source
        else:
            if acc is not None:
                newchildren.append(acc)
                acc = None
            newchildren.append(n)

    if acc is not None:
        newchildren.append(acc)
    node.children = newchildren


def compile_intermediate(parse_result):
    dom = make_dom(parse_result)
    strip_whitespace(dom, False)
    strip_spaces_between_directives(dom)
    combine_entities(dom)
    compiled = _compile_node(dom)
    _strip_unwanted_nodes(compiled)
    _concatentate_adjacent_text_nodes(compiled)
    _postprocess(compiled)
    return compiled


def _strip_unwanted_nodes(root):
    for parent, node, pos in im.walk_tree_post(root):
        if not parent:
            continue
        if node.__class__ is im.ContainerNode:
            parent.children[pos : pos + 1] = node.children

        elif node.__class__ is im.TextNode and node.content == "":
            parent.children[pos : pos + 1] = []

        elif node.__class__ is im.NullNode:
            parent.children[pos : pos + 1] = []


def _concatentate_adjacent_text_nodes(root):
    for parent, node, pos in im.walk_tree_post(root):
        if not isinstance(node, im.TextNode):
            continue
        try:
            sibling = parent.children[pos + 1]
        except IndexError:
            continue
        if isinstance(sibling, im.TextNode):
            node.content = "".join([node.content, sibling.content])
            parent.children[pos : pos + 2] = [node]


def _postprocess(root):
    postprocessors = {ElsePlaceholder: [pp_move_else_inside_if]}
    for parent, item, idx in im.walk_tree_post(root):
        if item.__class__ in postprocessors:
            for p in postprocessors[item.__class__]:
                p(item, parent)


def pp_move_else_inside_if(elsenode, parent):
    siblings_before = reversed(list(enumerate(parent.children)))
    list(takewhile(lambda item: item[1] is not elsenode, siblings_before))

    for ix, item in siblings_before:
        if isinstance(item, im.IfNode):
            item.else_ = im.ElseNode(children=elsenode.children)
            break
    else:
        raise PigletParseError("py:else without an py:if", lineno=elsenode.pos.line)

    parent.children.remove(elsenode)


def _compile_node(node):
    """
    Compile a single node to intermediate representation
    """

    if isinstance(node, Fragment):
        root = im.ContainerNode.factory()
        root.extend(map(_compile_node, node.children))
        return root

    elif isinstance(node, parse.Text):
        return _compile_text(node)

    elif isinstance(node, Element):
        directives, remaining = get_directives(node)
        if directives:
            return _compile_element_with_directives(node, directives, remaining)

        else:
            return _compile_vanilla_element(node)

    elif isinstance(node, parse.PI):
        if node.target == "python":
            return im.InlineCodeNode(pysrc=textwrap.dedent(node.content), pos=node.pos)
        else:
            return im.TextNode.factory(node.source, pos=node.pos)

    elif isinstance(node, parse.Entity):
        return im.TextNode.factory(node.source, pos=node.pos)

    elif isinstance(node, parse.Comment):
        c = "<!--" + node.content + "-->"
        n = im.TextNode.factory(c, pos=node.pos)
        if node.content.startswith("!"):
            n.content = ""
        return n

    elif isinstance(node, parse.Declaration):
        return im.TextNode.factory(node.source, pos=node.pos)

    elif isinstance(node, parse.CDATA):
        return im.TextNode.factory(node.source, pos=node.pos, autoescape=False)

    else:
        raise PigletParseError("Unknown node type: {!r}".format(node))


def _compile_text(node):
    """
    Compile :class:`piglet.parse.Text` object ``node`` into intermediate Text
    and Interpolation nodes.

    :param node: the :class:`piglet.parse.Text` object
    """
    items = interpolate.parse_interpolations(node.content)
    container = im.ContainerNode()
    pos = node.pos
    for i in items:
        if isinstance(i, (str, ustr)):
            container.append(im.TextNode(i, pos=pos))
            pos = pos.advance(i)
        elif isinstance(i, interpolate.Interpolation):
            container.append(
                im.InterpolateNode.factory(
                    value=unescape(i.value),
                    pos=pos,
                    autoescape=i.autoescape and (not node.cdata),
                )
            )
            pos = pos.advance(i.source)
    return container


def _compile_open_tag(node, tagname_expr=None, extra_attrs=None, strip_condition=None):
    """
    Compile the open tag attributes.
    This needs special casing when interpolations are present, eg in the case:

        <option selected="1 if item == 'foo' else None">

    The selected attribute should be entirely omitted if the expression
    evaluates to None.

    :param node: the parse.Element node to compile
    :param tagname_expr: a python expression to replace the tag name
    :param extra_attrs: A list of python source expressions, each returning a
                        list of attributes
    :param strip_condition: a python boolean expression to decide whether
                            to output the open tag at all
    """
    if isinstance(node, Fragment) and not tagname_expr:
        return im.ContainerNode(pos=node.pos)

    container = wn = im.ContainerNode(pos=node.pos)

    if strip_condition:
        if strip_condition in {"True", "1"}:
            container.append(im.NullNode(pos=node.pos))
        else:
            container.append(
                im.IfNode(
                    test="not ({})".format(strip_condition), children=[], pos=node.pos
                )
            )
        wn = container.tip()

    wn.append(im.TextNode.factory("<", pos=node.pos))
    if tagname_expr:
        wn.append(im.InterpolateNode(value=tagname_expr, pos=node.end_pos))
    else:
        wn.append(im.TextNode.factory(node.qname, pos=node.pos))
    wn.append(im.TextNode.factory(node.space, pos=node.pos))
    if extra_attrs:
        wn.append(im.InlineCodeNode(pysrc="__piglet_attrs = {}", pos=node.pos))
        for item in extra_attrs:
            wn.append(
                im.InlineCodeNode(
                    pysrc="__piglet_attrs.update({})".format(item), pos=node.pos
                )
            )

    for a in node.attrs.values():
        if node.attrs and extra_attrs:
            wn.append(
                im.IfNode(test="'{}' not in __piglet_attrs".format(a.name), pos=a.pos)
            )
            _compile_attr(a, wn.tip())
        else:
            _compile_attr(a, wn)

    if extra_attrs:
        for_ = im.ForNode(
            each=("__piglet_attr_k, __piglet_attr_v " "in __piglet_attrs.items()"),
            pos=node.end_pos,
            children=[],
        )
        if_ = im.IfNode(
            test="__piglet_attr_v is not None",
            pos=node.end_pos,
            children=[
                im.TextNode(" ", pos=node.end_pos),
                im.InterpolateNode(value="__piglet_attr_k", pos=node.end_pos),
                im.TextNode('="', pos=node.end_pos),
                im.InterpolateNode(value="__piglet_attr_v", pos=node.end_pos),
                im.TextNode('"', pos=node.end_pos),
            ],
        )
        for_.children.append(if_)
        wn.append(for_)
    wn.append(im.TextNode(content=node.end, pos=node.end_pos))
    return container


def _compile_attr(a, container):
    """
    Compile a single parse.Attribute
    """
    intro_text = im.TextNode.factory(
        "{0.name}{0.space1}={0.space2}{0.quote}".format(a), pos=a.pos
    )
    outro_text = im.TextNode.factory(
        "{0.quote}{0.space3}".format(a), pos=a.value_pos.advance(a.value)
    )

    items = interpolate.parse_interpolations(a.value)

    if (
        a.name in HTML_EMPTY_ATTRS
        and len(items) == 1
        and isinstance(items[0], interpolate.Interpolation)
    ):
        interp = items[0]
        container.append(
            im.WithNode(
                vars=[("__piglet_tmp", interp.value)],
                pos=a.pos,
                children=[
                    im.IfNode(
                        test="__piglet_tmp is not None",
                        pos=a.pos,
                        children=[
                            intro_text,
                            im.InterpolateNode(value="__piglet_tmp", pos=a.value_pos),
                            outro_text,
                        ],
                    )
                ],
            )
        )
    else:
        container.append(intro_text)
        pos = a.value_pos
        for i in items:
            if isinstance(i, (str, ustr)):
                container.append(im.TextNode.factory(i, pos))
                pos = pos.advance(i)
            else:
                container.append(im.InterpolateNode(value=unescape(i.value), pos=pos))
                pos = pos.advance(i.source)
        outro_text.pos = pos
        container.append(outro_text)


def _compile_close_tag(node, tagname_expr=None, strip_condition=None):
    if not node.close_tag or (isinstance(node, Fragment) and not tagname_expr):
        return im.TextNode("", pos=node.close_tag_pos)

    if tagname_expr:
        closetag = im.ContainerNode(
            children=[
                im.TextNode.factory("</", pos=node.close_tag_pos),
                im.InterpolateNode(value=tagname_expr, pos=node.close_tag_pos),
                im.TextNode.factory(">", pos=node.close_tag_pos),
            ]
        )
    else:
        closetag = im.TextNode(node.close_tag, pos=node.close_tag_pos)
    if strip_condition:
        if strip_condition in {"True", "1"}:
            return im.NullNode(children=[closetag])
        else:
            return im.IfNode(
                test="not ({})".format(strip_condition),
                children=[closetag],
                pos=node.close_tag_pos,
            )
    return closetag


def _compile_vanilla_element(node):
    container = im.ContainerNode()
    container.append(_compile_open_tag(node))
    container.extend(map(_compile_node, node.children))
    container.append(_compile_close_tag(node))
    return container


def append_directive_nodes(srcnode, directives, container):
    working_node = container
    for directive, data, pos in directives:
        if directive.factory:
            child = directive.factory(**data)
            child.pos = pos
            for _, item, _ in im.walk_tree_pre(child):
                item.pos = pos
            working_node.children.append(child)
            working_node = child
    return working_node


def _compile_element_with_directives(orig, directives, remaining):

    container = working_node = im.ContainerNode()
    inner_directives = [(d, dd, pos) for d, dd, pos in directives if d.inner]
    outer_directives = [(d, dd, pos) for d, dd, pos in directives if not d.inner]

    # Add outer directives
    working_node = append_directive_nodes(orig, outer_directives, container)
    if not isinstance(working_node, im.ContainerNode):
        working_node = im.NullNode()
        container.append(working_node)

    # Add tag
    extra_attrs = []
    strip_condition = None
    for d, dd, pos in directives:
        extras = d.get_extra_attrs(dd, remaining)
        if extras:
            extra_attrs.append(extras)
        strip_condition = d.get_strip_condition(dd, remaining) or strip_condition
        tagname_expr = d.get_tagname_expr(dd, remaining)

    working_node.append(
        _compile_open_tag(
            remaining,
            tagname_expr=tagname_expr,
            strip_condition=strip_condition,
            extra_attrs=extra_attrs,
        )
    )

    # Add inner directives
    inner_working_node = append_directive_nodes(orig, inner_directives, working_node)

    if not isinstance(inner_working_node, im.ContainerNode):
        inner_working_node = im.NullNode()
        working_node.append(inner_working_node)

    inner_working_node.extend(map(_compile_node, remaining.children))

    working_node.append(
        _compile_close_tag(
            remaining, tagname_expr=tagname_expr, strip_condition=strip_condition
        )
    )

    return container
