# encoding=utf-8
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

from piglet import compilexml as cx
from piglet import intermediate as im
from piglet.parse import parse_html
from piglet.exceptions import PigletParseError
from piglet.position import Position as P


def c(s):
    return cx.compile_intermediate(parse_html(s)).children


def test_it_compiles_element():
    assert c('<a href="foo">bar</a>') == [
        im.TextNode(pos=P(1, 1), content='<a href="foo">bar</a>'),
    ]


def test_it_compiles_entity():
    assert c("&amp;") == [im.TextNode(pos=P(1, 1), content="&amp;")]


def test_it_compiles_block():
    assert c('<py:block name="foo">bar</py:block>') == [
        im.BlockNode(
            pos=P(1, 1),
            name="foo",
            children=[im.TextNode(pos=P(1, 22), content="bar"),],
        )
    ]


def test_it_compiles_tag_with_inner_directive_attrs():
    assert c('<div py:block="foo">bar</div>') == [
        im.TextNode(pos=P(1, 1), content="<div>"),
        im.BlockNode(
            pos=P(1, 16),
            name="foo",
            children=[im.TextNode(pos=P(1, 21), content="bar")],
        ),
        im.TextNode(content="</div>", pos=P(1, 24)),
    ]


def test_it_compiles_tag_with_outer_directive_attrs():
    assert c('<div py:if="foo">bar</div>') == [
        im.IfNode(
            pos=P(1, 13),
            test="foo",
            children=[im.TextNode(pos=P(1, 1), content="<div>bar</div>"),],
        ),
    ]


def test_it_compiles_choose():

    assert c(
        '<py:choose test="foo">'
        '<py:when test="1">one</py:when>'
        '<py:when test="2">two</py:when>'
        "<p>some text</p>"
        "<py:otherwise>zzz</py:otherwise>"
        "</py:choose>"
    ) == [
        im.ChooseNode(
            pos=P(1, 1),
            test="foo",
            children=[
                im.WhenNode(
                    pos=P(1, 23),
                    test="1",
                    children=[im.TextNode(pos=P(1, 41), content="one")],
                ),
                im.WhenNode(
                    pos=P(1, 54),
                    test="2",
                    children=[im.TextNode(pos=P(1, 72), content="two")],
                ),
                im.TextNode(pos=P(1, 85), content="<p>some text</p>"),
                im.OtherwiseNode(
                    pos=P(1, 101), children=[im.TextNode(pos=P(1, 115), content="zzz")]
                ),
            ],
        )
    ]


def test_it_compiles_if_else():

    assert c('<py:if test="foo">xxx</py:if>' "  " "<py:else>zzz</py:else>") == [
        im.IfNode(
            pos=P(1, 1),
            test="foo",
            children=[im.TextNode(pos=P(1, 19), content="xxx")],
            else_=im.ElseNode(children=[im.TextNode(pos=P(1, 41), content="zzz")]),
        )
    ]


def test_it_compiles_pycall():
    assert c(
        '<py:call function="func(1)">'
        '<py:keyword name="x">foo</py:keyword>'
        '<py:keyword name="y">bar</py:keyword>'
        "</py:call>"
    ) == [
        im.Call(
            pos=P(1, 1),
            function="func(1)",
            children=[
                im.CallKeyword(
                    pos=P(1, 29),
                    name="x",
                    children=[im.TextNode(pos=P(1, 50), content="foo")],
                ),
                im.CallKeyword(
                    pos=P(1, 66),
                    name="y",
                    children=[im.TextNode(pos=P(1, 87), content="bar")],
                ),
            ],
        )
    ]


def test_it_compiles_pyimport():
    assert c('<py:import href="foo.html" alias="bar"/>') == [
        im.ImportNode(pos=P(1, 1), href="foo.html", alias="bar")
    ]


def test_it_raises_sensible_exceptions_on_malformed_directives():
    try:
        # Use the Template interface as it allows us to check that the filename
        # is included in the error message
        from piglet.template import HTMLTemplate

        HTMLTemplate("\n\n<py:wtf/>", "test.html")
    except PigletParseError as e:
        message = str(e)
        assert message == (
            "Unrecognized directive 'py:wtf' in element "
            "<py:wtf/> in test.html, line 3"
        )
    else:
        assert False


def test_it_compiles_replace():
    assert c('<py:replace value="foo">bar</py:replace>') == [
        im.InterpolateNode(pos=P(1, 1), value="foo")
    ]
    assert c('<img py:replace="img_tag()"/>') == [
        im.InterpolateNode(pos=P(1, 18), value="img_tag()")
    ]


def test_it_compiles_content():
    assert c('<div py:content="body">foo</div>') == [
        im.TextNode(pos=P(1, 1), content="<div>"),
        im.InterpolateNode(pos=P(1, 18), value="body"),
        im.TextNode(pos=P(1, 27), content="</div>"),
    ]


def test_it_compiles_inline_interpolation():
    assert c("<div>$a</div>") == [
        im.TextNode(pos=P(1, 1), content="<div>"),
        im.InterpolateNode(pos=P(1, 6), value="a"),
        im.TextNode(pos=P(1, 8), content="</div>"),
    ]


def test_it_compiles_conditional_attr_interpolation():

    assert c('<input selected="$foo"/>') == [
        im.TextNode(pos=P(1, 1), content="<input "),
        im.WithNode(
            pos=P(1, 8),
            vars=[("__piglet_tmp", "foo")],
            children=[
                im.IfNode(
                    pos=P(1, 8),
                    test="__piglet_tmp is not None",
                    children=[
                        im.TextNode(pos=P(1, 8), content='selected="'),
                        im.InterpolateNode(pos=P(1, 18), value="__piglet_tmp"),
                        im.TextNode(pos=P(1, 22), content='"'),
                    ],
                )
            ],
        ),
        im.TextNode(pos=P(1, 23), content="/>"),
    ]


def test_it_compiles_interpolations_with_entities():
    assert c("${1 &gt; 0}") == [im.InterpolateNode(pos=P(1, 1), value="1 > 0")]


def test_it_compiles_regular_attr_interpolation():

    assert c('<div class="x $foo"></div>') == [
        im.TextNode(pos=P(1, 1), content='<div class="x '),
        im.InterpolateNode(pos=P(1, 15), value="foo"),
        im.TextNode(pos=P(1, 19), content='"></div>'),
    ]


def test_it_compiles_pystrip():
    assert c('<div py:strip=""><p>foo</p></div>') == [
        im.TextNode(pos=P(1, 18), content="<p>foo</p>"),
    ]
    assert c('<div py:strip="x"><p>foo</p></div>') == [
        im.IfNode(
            pos=P(1, 1),
            test="not (x)",
            children=[im.TextNode(pos=P(1, 1), content="<div>")],
        ),
        im.TextNode(pos=P(1, 19), content="<p>foo</p>"),
        im.IfNode(
            pos=P(1, 29),
            test="not (x)",
            children=[im.TextNode(pos=P(1, 29), content="</div>")],
        ),
    ]


def test_it_processes_strip_whitespace():
    assert c(u'<p py:whitespace="strip"> <b> foo bar </b> </p>') == [
        im.TextNode(pos=P(1, 1), content=u"<p><b>foo bar</b></p>"),
    ]

    assert c(u'<p py:whitespace="strip">one <b>and</b> two <b>!</b></p>') == [
        im.TextNode(pos=P(1, 1), content=u"<p>one <b>and</b> two <b>!</b></p>"),
    ]

    assert c(u'<p py:whitespace="strip">\n<b> foo bar </b>\n</p>') == [
        im.TextNode(pos=P(1, 1), content=u"<p><b>foo bar</b></p>"),
    ]

    assert c(u'<p py:whitespace="strip"> <b>\u00a0 foo bar </b> </p>') == [
        im.TextNode(pos=P(1, 1), content=u"<p><b>\u00a0 foo bar</b></p>"),
    ]

    assert c(
        u'<p py:whitespace="strip"> ' u'<b py:whitespace="preserve"> foo </b></p>'
    ) == [im.TextNode(pos=P(1, 1), content=u"<p><b> foo </b></p>"),]


def test_it_processes_strip_whitespace_and_keeps_line_numbering():
    assert c(
        u'<p py:whitespace="strip"> ' u'<py:if test="True"> hello! </py:if></p>'
    ) == [
        im.TextNode(pos=P(1, 1), content="<p>"),
        im.IfNode(
            pos=P(1, 27),
            test="True",
            children=[im.TextNode(pos=P(1, 47), content="hello!")],
        ),
        im.TextNode(pos=P(1, 62), content="</p>"),
    ]
    assert c(
        u'<html><py:whitespace value="strip"> '
        u'<p><py:if test="True"> hello! </py:if></p>'
        u"</py:whitespace></html>"
    ) == [
        im.TextNode(pos=P(1, 1), content="<html><p>"),
        im.IfNode(
            pos=P(1, 40),
            test="True",
            children=[im.TextNode(pos=P(1, 60), content="hello!")],
        ),
        im.TextNode(pos=P(1, 75), content="</p></html>"),
    ]


def test_it_strips_whitespace_around_text():
    assert c('<p py:whitespace="strip"> foo </p>') == [
        im.TextNode(pos=P(1, 1), content="<p>foo</p>")
    ]


def test_it_strips_whitespace_inside_tags():
    assert c('<p py:whitespace="strip"> <a/> </p>') == [
        im.TextNode(pos=P(1, 1), content="<p><a/></p>")
    ]


def test_it_strips_whitespace_between_elements():
    assert c('<p py:whitespace="strip"> \n <a/>\n<b/> \n </p>') == [
        im.TextNode(pos=P(1, 1), content="<p><a/><b/></p>")
    ]


def test_it_strips_space_between_py_directives():
    assert c('<py:if test="0"></py:if> <py:if test="1"></py:if>') == [
        im.IfNode(pos=P(1, 1), children=[], test="0"),
        im.IfNode(pos=P(1, 26), children=[], test="1"),
    ]


def test_it_preserves_ordering_of_namespaced_attributes():
    assert c('<html lang="en" xml:lang="en"></html>') == [
        im.TextNode(
            pos=P(line=1, char=1), content='<html lang="en" xml:lang="en"></html>'
        )
    ]
