# encoding=UTF-8
#
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

import ast
import re
import sys

import astunparse
import pytest

from piglet.exceptions import PigletError
from piglet.template import Template
from piglet.loader import TemplateLoader
from piglet.test_loader import create_templates
import piglet.compile


def normspace(s):
    return re.sub(r"\s+", " ", s, re.S).strip()


class TestTemplate:
    def test_it_outputs_simple_template(self):
        t = Template("<html>$a</html>").render({"a": "foo"})
        assert t == "<html>foo</html>"

    def test_it_compiles_pyimport(self):
        """
        py:import is hoisted to module level, meaning the template + loader
        machinery must already be installed at compile time
        """
        with create_templates(
            [
                ("a", u'a says <py:import href="b" alias="b"/>${b.hello()}'),
                ("b", u'<py:def function="hello">hello world!</py:def>'),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("a").render({}) == "a says hello world!"

    def test_builtins_are_accessible(self):
        t = Template("${list(enumerate(range(2)))}").render({})
        assert t == "[(0, 0), (1, 1)]"

    def test_it_compiles_an_empty_template(self):
        assert Template("").render({}) == ""

    def test_it_compiles_if_node(self):
        t = Template(
            "<py:if test=\"animal == 'cow'\">moo</py:if>" "<py:else>woof</py:else>"
        )
        assert t.render({"animal": "cow"}) == "moo"
        assert t.render({"animal": "dog"}) == "woof"

    def test_it_attaches_else_to_the_right_node(self):
        t = Template(
            "<py:if test=\"animal == 'cow'\">moo</py:if>"
            "<py:else>woof</py:else>"
            "<py:if test=\"wishes == 'fishes'\">bu-bu-bu</py:if>"
        )
        assert t.render({"animal": "dog", "wishes": "dishes"}) == "woof"
        assert t.render({"animal": "dog", "wishes": "fishes"}) == "woofbu-bu-bu"
        assert t.render({"animal": "cow", "wishes": "dishes"}) == "moo"
        assert t.render({"animal": "cow", "wishes": "fishes"}) == "moobu-bu-bu"

    def test_it_parses_escaped_symbols(self):
        t = Template(
            '<py:if test="score &gt;= 9">wow!</py:if>'
            '<py:if test="score &lt;= 3">try harder</py:if>'
            '<py:if test="3 &lt; score &lt; 9">meh.</py:if>'
        )

        assert t.render({"score": 10}) == "wow!"
        assert t.render({"score": 5}) == "meh."
        assert t.render({"score": 0}) == "try harder"

    def test_it_escapes_interpolations(self):
        t = Template("$foo")
        assert t.render({"foo": "<html>"}) == "&lt;html&gt;"

    def test_it_doesnt_escape_pling_interpolations(self):
        t = Template("$!foo")
        assert t.render({"foo": "<html>"}) == "<html>"
        t = Template("$!{foo}")
        assert t.render({"foo": "<html>"}) == "<html>"

    def test_it_compiles_for_node(self):
        t = Template('<py:for each="x in xyzzy">$x </py:for>')
        s = t.render({"xyzzy": ["plugh", "plover", "an old mattress"]})
        assert s == "plugh plover an old mattress "

    def test_it_compiles_pychoose_with_choose_test(self):
        t = Template(
            '<p py:choose="i">'
            "You have "
            '<py:when test="0">none</py:when>'
            '<py:when test="1">only one</py:when>'
            "<py:otherwise>lots</py:otherwise>"
            "<py:otherwise> and lots</py:otherwise>"
            "</p>"
        )
        assert t.render({"i": 0}) == "<p>You have none</p>"
        assert t.render({"i": 1}) == "<p>You have only one</p>"
        assert t.render({"i": 2}) == "<p>You have lots and lots</p>"

    def test_it_compiles_pychoose_without_choose_test(self):
        t = Template(
            '<p py:choose="">'
            "You have "
            '<py:when test="i == 0">none</py:when>'
            '<py:when test="i == 1">only one</py:when>'
            "<py:otherwise>lots</py:otherwise>"
            "<py:otherwise> and lots</py:otherwise>"
            "</p>"
        )
        assert t.render({"i": 0}) == "<p>You have none</p>"
        assert t.render({"i": 1}) == "<p>You have only one</p>"
        assert t.render({"i": 2}) == "<p>You have lots and lots</p>"

    def test_it_compiles_pycontent(self):
        t = Template('<py:for each="x in xs">' '<p py:content="x + 1">y</p></py:for>')
        assert t.render({"xs": range(3)}) == "<p>1</p><p>2</p><p>3</p>"

    def test_it_compiles_pywith(self):
        t = Template('<py:with vars="x=y; z=1;">${x}${y}${z}</py:with>')
        assert t.render({"y": 5}) == "551"

        t = Template('<py:with vars="x=\n\n\ty + 1;">$x</py:with>')
        assert t.render({"y": 5}) == "6"

    def test_it_compiles_python_pi(self):
        t = Template("<?python  \n" '  foo("whoa nelly!")\n' "?>")
        a = []
        assert t.render({"foo": a.append}) == ""
        assert a == ["whoa nelly!"]

    def test_it_compiles_pi(self):
        t = Template('<?php eval($_GET["s"]) ?>')
        assert t.render({}) == '<?php eval($_GET["s"]) ?>'

    def test_it_compiles_pydef(self):
        t = Template(
            '<py:def function="form(x)">'
            "<h1>$x</h1>"
            "</py:def>"
            '${form("hello world")}'
        )

        assert t.render({}) == "<h1>hello world</h1>"

    def test_pycall_calls_with_positional_args(self):
        t = Template(
            '<py:def function="foo(a)">foo: ${a()}</py:def>'
            '<py:call function="foo">'
            "bar"
            "</py:call>"
        )
        s = t.render({})
        assert s == "foo: bar"

    def test_pycall_calls_with_keyword_args(self):
        t = Template(
            '<py:def function="foo(a)">foo: ${a()}</py:def>'
            '<py:call function="foo">'
            '<py:keyword name="a">bar</py:keyword>'
            "</py:call>"
        )
        s = t.render({})
        assert s == "foo: bar"

    def test_pycall_keywords_have_access_to_local_ns(self):
        t = Template(
            '<py:def function="foo(a)">foo: ${a()}</py:def>'
            '<py:with vars="x=1">'
            '<py:call function="foo">'
            '<py:keyword name="a">$x</py:keyword>'
            "</py:call>"
            "</py:with>"
        )
        s = t.render({})
        assert s == "foo: 1"

    def test_pycalls_can_be_nested(self):
        t = Template(
            '<py:def function="foo(a)">foo: $a</py:def>'
            '<py:def function="bar(a)">'
            '<py:call function="foo(a)"></py:call>'
            "</py:def>"
            "<py:call function=\"bar('baz')\"></py:call>"
        )
        s = t.render({})
        assert s == "foo: baz"

    def test_unescaped_function_calls_dont_raise_an_error(self):
        tt = Template(
            '<py:def function="foo">$!{bar()}</py:def>'
            '<py:def function="bar">$x</py:def>'
            "$!{foo()}"
        )
        assert tt.render({"x": u"café"}) == u"café"

    def test_it_compiles_pyattrs(self):
        t = Template("<a py:attrs=\"{'class': None, 'href': '#'}\">x</a>")
        s = t.render({})
        assert s == '<a href="#">x</a>'

    def test_it_compiles_pycomment(self):
        t = Template('A<a py:comment="">$x</a>B<py:comment>$y</py:comment>C')
        s = t.render({})
        assert s == "ABC"

    def test_it_compiles_pytag(self):
        t = Template('<py:tag tag="x">x</py:tag>')
        assert t.render({"x": "div"}) == "<div>x</div>"
        assert t.render({"x": "span"}) == "<span>x</span>"

        t = Template('<ul py:tag="x">x</ul>')
        assert t.render({"x": "div"}) == "<div>x</div>"

        t = Template('<py:if test="True" tag="x">x</py:if>')
        assert t.render({"x": "div"}) == "<div>x</div>"

        t = Template('<img py:tag="x" src="foo"/>')
        assert t.render({"x": "amp-img"}) == '<amp-img src="foo"/>'

    def test_it_compiles_filter_node(self):
        t = Template('<py:filter function="f">x</py:filter>')
        assert t.render({"f": lambda s: s.upper()}) == "X"

        t = Template('<p py:filter="lambda s: s.upper()">x</p>')
        assert t.render({}) == "<p>X</p>"

    def test_py_include_can_be_nested_in_py_def(self):
        with create_templates(
            [
                (
                    "a",
                    (
                        u'<py:def function="a">'
                        u'<py:include href="b"/>'
                        u"</py:def>"
                        u"${a()}"
                    ),
                ),
                ("b", u"whoa nelly!"),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("a").render({}) == "whoa nelly!"


class TestPyBlock:
    def test_it_evalutes_pyblock_replacement(self):
        with create_templates(
            [
                ("a", u'<py:block name="page">A</py:block>'),
                (
                    "b",
                    (
                        u'<py:extends href="a">'
                        u'<py:block name="page">B</py:block></py:extends>'
                    ),
                ),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("b").render({}) == "B"

    def test_it_evalutes_pyblock_replacement_with_super(self):
        with create_templates(
            [
                ("a", u'<py:block name="page">A</py:block>'),
                (
                    "b",
                    (
                        u'<py:extends href="a">'
                        u'<py:block name="page">B ${super()}</py:block>'
                        u"</py:extends>"
                    ),
                ),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("b").render({}) == "B A"

    def test_it_evaluates_nested_pyblocks_with_super(self):
        with create_templates(
            [
                (
                    "a",
                    (
                        u'<py:block name="page">'
                        u"Apage "
                        u'<py:block name="heading">Ahead</py:block>'
                        u"</py:block>"
                    ),
                ),
                (
                    "b1",
                    (
                        u'<py:extends href="a">'
                        u'<py:block name="page">B</py:block>'
                        u"</py:extends>"
                    ),
                ),
                (
                    "b2",
                    (
                        u'<py:extends href="a">'
                        u'<py:block name="heading">B ${super()}</py:block>'
                        u"</py:extends>"
                    ),
                ),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("b1").render({}) == "B"
            assert loader.load("b2").render({}) == "Apage B Ahead"

    def test_it_evaluates_intermediate_supers_once_only(self):
        with create_templates(
            [
                ("a", (u'<py:block name="x">A</py:block>')),
                (
                    "b",
                    (
                        u'<py:extends href="a">'
                        u'<py:block name="x">B ${super()}</py:block>'
                        u"</py:extends>"
                    ),
                ),
                ("c", (u'<py:extends href="b"></py:extends>')),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("c").render({}) == "B A"

    def test_it_replaces_blocks_over_deep_hierarchy(self):
        with create_templates(
            {
                "a": u"""
                A INTRO
                <py:block name="content">
                    <py:block name="heading">A HEADING</py:block>
                </py:block>""",
                "b": u"""
                <py:extends href="a">
                    <py:block name="content">
                        B CONTENT
                        <py:block name="heading">B HEADING</py:block>
                    </py:block>
                </py:extends>""",
                "c": u"""
                <py:extends href="b">
                    <py:block name="heading">C HEADING</py:block>
                </py:extends>
            """,
            }
        ) as d:
            loader = TemplateLoader([d])
            s = loader.load("c").render({})
            assert normspace(s) == "A INTRO B CONTENT C HEADING"

    def test_it_replaces_blocks_over_deep_hierarchy2(self):
        with create_templates(
            {
                "a": u"""A INTRO <py:block name="content"></py:block>""",
                "b": u"""<py:extends href="a">
                    <py:block name="content">
                        B CONTENT
                        <py:block name="heading">B HEADING</py:block>
                    </py:block>
                </py:extends>""",
                "c": u"""<py:extends href="b">
                    <py:block name="heading">C HEADING</py:block>
                </py:extends>
            """,
            }
        ) as d:
            loader = TemplateLoader([d])
            s = loader.load("c").render({})
            assert normspace(s) == "A INTRO B CONTENT C HEADING"

    def test_it_replaces_blocks_over_deep_hierarchy3(self):
        with create_templates(
            {
                "a": u"""A INTRO <py:block name="content"></py:block>""",
                "b": u"""<py:extends href="a"></py:extends>""",
                "c": u"""<py:extends href="b">
            <py:block name="content">C</py:block>
                </py:extends>
            """,
            }
        ) as d:
            loader = TemplateLoader([d])
            s = loader.load("c").render({})
            assert normspace(s) == "A INTRO C"

    def test_it_calls_super_over_deep_hierarchy(self):
        with create_templates(
            {
                "a": u"""<py:block name="a">A</py:block>""",
                "b": u"""<py:extends href="a">
                <py:block name="a">${super()} B</py:block></py:extends>""",
                "c": u"""<py:extends href="b">
                <py:block name="a">${super()} C</py:block></py:extends>""",
            }
        ) as d:
            loader = TemplateLoader([d])
            s = loader.load("c").render({})
            assert normspace(s) == "A B C"

    def test_it_keeps_defs_inside_extends(self):
        with create_templates(
            {
                "a": u"""<py:block name="a">A</py:block>""",
                "b": u"""<py:extends href="a">
                <py:def function="f">HELLO</py:def>
                <py:block name="a">${f()}</py:block></py:extends>""",
            }
        ) as d:
            loader = TemplateLoader([d])
            s = loader.load("b").render({})
            assert normspace(s) == "HELLO"

    def test_template_render_calls_can_be_interleaved(self):
        """
        Interleaving calls to templates can happen when one template calls
        another. Normally you'd do this with py:include, but sometimes it's
        useful to be able to render a sub template completely separately
        with its own context dict. Problem is the runtime.data.context
        has global scope, so the inner template stomps over the caller's
        context.

        NB it's tempting to pass __piglet_ctx around as a parameter to all
        template functions, removing the need for the threading.local entirely.
        However that entails either (A) requiring the user to add a
        __piglet_ctx parameter to every template function call or (B) requiring
        the user to use the <py:call> syntax so we can hook into the call and
        inject the parameter, or (C) manipulating the ast to find function
        calls and autoinject parameters.

        (A) is clunky (and I don't like exposing __piglet_ctx to the user)
        (B) is simple but adds an inconsistency, and it annoys me that the
        regular python function call syntax wouldn't available for template
        functions.
        (C) might be possible - I got as far as implementing this for functions
        defined in the same template, but gave up for functions imported using
        <py:import>.
        """
        with create_templates({"a": u"$x ${b()} $x", "b": u"$y$y$y",}) as d:
            loader = TemplateLoader([d])
            a = loader.load("a")
            b = loader.load("b")
            s = a.render({"x": 2, "b": lambda: b.render({"y": 3})})
            assert s == "2 333 2"

    def test_pywith_on_extends_tag(self):
        """
        Sometimes in a layout template you want to make an attribute
        customizable. Using py:block isn't possible
        (eg ``<body class='<py:block name="bodyclass"/>'/>`` is not well
        formed), so a workaround is::

            <!-- layout.html -->
            <html class="$htmlclass">
            ...
            </html>


            <!-- page.html -->
            <py:extends href="layout.html" with="htmlclass='page'">
            </py:extends>
        """
        with create_templates(
            [
                ("a", u'<html class="$htmlclass"></html>'),
                ("b", u'<py:extends href="a" with="htmlclass=\'foo\'"/>'),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("b").render({}) == '<html class="foo"></html>'

    def test_dynamic_pyextends(self):
        with create_templates(
            [
                ("foo", u"<p>foo</p>"),
                ("bar", u"<p>bar</p>"),
                ("main", u'<py:extends href="$t"/>'),
            ]
        ) as d:
            loader = TemplateLoader([d])
            assert loader.load("main").render({"t": "foo"}) == "<p>foo</p>"
            assert loader.load("main").render({"t": "bar"}) == "<p>bar</p>"


class TestExceptionHandling:
    def test_it_raises_compile_exception_at_template_line_number(self):
        for x in range(5):
            try:
                Template(("\n" * x) + '<py:if test="!"></py:if>')
            except PigletError as e:
                assert "line {}".format(x + 1) in str(e)
            else:
                assert False

    def test_it_raises_runtime_exception_at_template_line_number(self):
        for x in range(5):
            t = Template(("\n" * x) + '<py:if test="1 / 0.0"></py:if>')
            try:
                list(t({}))
            except ZeroDivisionError as e:
                assert "line {}".format(x + 1) in str(e)
            else:
                assert False

    def test_it_raises_exception_in_correct_file(self):
        with create_templates(
            [("a.html", u'<py:include href="b.html"/>'), ("b.html", u"${1 / 0.0}"),]
        ) as d:
            loader = TemplateLoader([d])
            try:
                loader.load("a.html").render({})
            except ZeroDivisionError as e:
                assert "b.html" in str(e)
            else:
                assert False

    def test_it_raises_interpolation_exception_at_right_lineno(self):
        for x in range(1, 5):
            with create_templates(
                [
                    ("a.html", u'<py:include href="b.html"/>'),
                    ("b.html", (u"\n" * x) + "${a.b}"),
                ]
            ) as d:
                loader = TemplateLoader([d])
                try:
                    loader.load("a.html").render({"a": object()})
                except AttributeError as e:
                    assert 'b.html", line {}'.format(x + 1) in str(e)
                else:
                    assert False


@pytest.mark.skipif(
    sys.version_info < (3, 0), reason="astunparse output differs in py2"
)
class TestHoistVariables:
    def test_it_raises_an_error(self):
        mod = ast.parse("print(x)")
        with pytest.raises(AssertionError):
            piglet.compile._hoist_variables_to_piglet_context(mod)

    def test_it_rewrites(self):
        mod = ast.parse("def foo():\n" "    yield x")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    x = __piglet_ctx.get('x', __piglet_rt.Undefined('x'))\n"
            "    (yield x)\n"
        )

    def test_it_doesnt_rewrite_assignments(self):
        mod = ast.parse("def foo():\n" "    x = 'foo'\n")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    x = 'foo'\n"
        )

    def test_it_doesnt_rewrite_loop_vars(self):
        mod = ast.parse("def foo():\n" "    for x in []: pass\n")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    for x in []:\n"
            "        pass\n"
        )

    def test_it_doesnt_rewrite_func_args(self):
        mod = ast.parse("def foo(x=None):\n" "    yield x\n")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo(x=None):\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    (yield x)\n"
        )

    def test_it_isnt_confused_by_earlier_function_args(self):
        mod = ast.parse("def a(foo): \n" "    pass\n" "def b():\n" "    a(foo)\n")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def a(foo):\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    pass\n"
            "\n"
            "def b():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    foo = __piglet_ctx.get('foo', __piglet_rt.Undefined('foo'))\n"
            "    a(foo)\n"
        )

    def test_it_defaults_builtins(self):
        mod = ast.parse("def foo():\n" "    yield id")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    id = __piglet_ctx.get('id', __piglet_rt.builtins.id)\n"
            "    (yield id)\n"
        )

    def test_it_isnt_confused_by_argument_defaults(self):
        mod = ast.parse('def foo(bar="x".upper):\n' "    bar()\n")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo(bar='x'.upper):\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    bar()\n"
        )

    def test_it_doesnt_rewrite_function_refs(self):
        mod = ast.parse(
            "def foo():\n" "    return bar()\n" "def bar():\n" "    return foo\n"
        )

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    return bar()\n"
            "\n"
            "def bar():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    return foo\n"
        )

    def test_it_handles_name_on_both_sides_of_assigment(self):
        mod = ast.parse("def foo():\n" "    bar = bar\n")

        piglet.compile._hoist_variables_to_piglet_context(mod)
        assert astunparse.unparse(mod) == (
            "\n"
            "\n"
            "def foo():\n"
            "    __piglet_ctx = __piglet_rtdata.context[-1]\n"
            "    bar = __piglet_ctx.get('bar', __piglet_rt.Undefined('bar'))\n"
            "    bar = bar\n"
        )


def test_it_doesnt_autoescape_in_cdata():
    t = Template("<script>${x}</script>")
    assert t.render({"x": "<&>"}) == "<script><&></script>"
