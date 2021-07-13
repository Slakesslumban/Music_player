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

import re

from hypothesis import given, strategies as s

from piglet import compilett, TemplateLoader, TextTemplate
from piglet.test_loader import create_templates


def toim(s):
    return compilett.compile_intermediate(compilett.parse_tt(s)).children


def test_it_compiles_if():
    tt = TextTemplate("{% if x > 1 %}yes{%end%}")
    assert tt.render({"x": 2}) == "yes"
    assert tt.render({"x": 1}) == ""


def test_it_compiles_quoted_if_conditions():
    tt = TextTemplate('{% if foo == "bar" %}yes{%end%}')
    assert tt.render({"foo": "bar"}) == "yes"
    assert tt.render({"foo": "baz"}) == ""


def test_it_compiles_if_else():
    tt = TextTemplate("{% if x > 1 %}yes{% else %}no{%end%}")
    assert tt.render({"x": 2}) == "yes"
    assert tt.render({"x": 1}) == "no"


def test_it_compiles_with():
    tt = TextTemplate('{% with x = "1"; y = "2"; z=y %}${x + y + z}{% end %}')
    assert tt.render({}) == "122"


def test_it_compiles_trans():
    n = toim("{% trans %}foo {% transname x %}$x{%end%}{% end %}")
    assert n[0].get_msgstr() == "foo ${x}"


def test_it_compiles_def():
    tt = TextTemplate('{% def foo(x) %}$x $x $x{%end%}${foo("a")}')
    assert tt.render({}) == "a a a"


def test_it_compiles_for():
    tt = TextTemplate("{% for x in xs %}$x! {%end%}")
    assert tt.render({"xs": [1, 2, 3]}) == "1! 2! 3! "


def test_it_doesnt_html_escape():
    tt = TextTemplate("$x")
    assert tt.render({"x": "&"}) == "&"


def test_it_strips_whitespace_when_directive_tag_alone_on_line():
    tt = TextTemplate("A \n\n {% if 1 %} \n B \n  {%end%} \n C\n")
    assert tt.render({}) == "A \n\n B \n C\n"


def test_it_preserves_whitespace_between_directives():
    tt = TextTemplate(
        "{% if 1 %}\n"
        "foo\n"
        "{% end %}\n"
        "\n"
        "{% if 2 %}\n"
        "bar\n"
        "{% end %}\n"
    )
    assert tt.render({}) == "foo\n\nbar\n"


@given(
    s.text(alphabet="\n \t"),
    s.text(alphabet="\n \t"),
    s.text(alphabet="\n \t"),
    s.text(alphabet="\n \t"),
    s.sampled_from(["", "-", "+"]),
    s.sampled_from(["", "-", "+"]),
    s.sampled_from(["", "-", "+"]),
    s.sampled_from(["", "-", "+"]),
)
def test_it_applies_whitespace_control(s1, s2, s3, s4, w1, w2, w3, w4):

    source = "A{s1}{{%{w1} if 1 {w2}%}}{s2}B{s3}{{%{w3}end{w4}%}}{s4}C".format(
        **locals()
    )
    tt = TextTemplate(source)

    expected = "A"
    if w1 == "+":
        expected += s1
    elif w1 == "":
        expected += re.sub(r"\n[ \t]*\Z", r"\n", s1)
    if w2 == "+":
        expected += s2
    elif w2 == "":
        expected += re.sub(r"\A[ \t]*\n", r"", s2)

    expected += "B"

    if w3 == "+":
        expected += s3
    elif w3 == "":
        expected += re.sub(r"\n[ \t]*\Z", r"\n", s3)
    if w4 == "+":
        expected += s4
    elif w4 == "":
        expected += re.sub(r"\A[ \t]*\n", r"", s4)

    expected += "C"
    result = tt.render({})
    assert result == expected


def test_it_compiles_extends_and_block():
    with create_templates(
        [
            ("a", u"A {% block foo %}foo{% end %}"),
            ("b", u'{% extends "a" %}{% block foo %}bar{% end %}{% end %}'),
        ]
    ) as d:
        loader = TemplateLoader([d], template_cls=TextTemplate)
        assert loader.load("a").render({}) == "A foo"
        assert loader.load("b").render({}) == "A bar"


def test_it_compiles_import():
    with create_templates(
        [
            ("a", u"A {% def afunc %}foo{% end %}"),
            ("b", u"B {% import a as A %}${A.afunc()}"),
        ]
    ) as d:
        loader = TemplateLoader([d], template_cls=TextTemplate)
        assert loader.load("b").render({}) == "B foo"
