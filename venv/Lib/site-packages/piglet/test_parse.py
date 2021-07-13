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

from collections import OrderedDict
from piglet.parse import (
    Attribute,
    OpenTag,
    CloseTag,
    Comment,
    Entity,
    PI,
    Text,
    parse_html,
)
from piglet.position import Position as P


class TestParser:
    def test_it_parses_element(self):
        assert parse_html("<p></p>") == [
            OpenTag(P(1, 1), "p", space=""),
            CloseTag(P(1, 4), "p"),
        ]
        assert parse_html("<p>a</p>") == [
            OpenTag(P(1, 1), "p", space=""),
            Text(P(1, 4), "a"),
            CloseTag(P(1, 5), "p"),
        ]

        assert parse_html("<html:p></html:p>") == [
            OpenTag(P(1, 1), "html:p", space=""),
            CloseTag(P(1, 9), "html:p"),
        ]

    def test_it_parses_attributes(self):
        assert parse_html('<p data-foo="bar"></p>') == [
            OpenTag(
                P(1, 1),
                "p",
                space=" ",
                attrs=OrderedDict(
                    [
                        (
                            "data-foo",
                            Attribute(
                                pos=P(1, 4),
                                name="data-foo",
                                value="bar",
                                value_pos=P(1, 14),
                                quote='"',
                                space1="",
                                space2="",
                                space3="",
                            ),
                        )
                    ]
                ),
            ),
            CloseTag(P(1, 19), "p"),
        ]

    def test_it_parses_comment(self):
        assert parse_html("<!-- x -->") == [Comment(P(1, 1), content=" x ")]

    def test_it_parses_pi(self):
        assert parse_html("<?php ?>") == [PI(P(1, 1), target="php", content=" ")]

    def test_it_parses_entity(self):
        assert parse_html("&nbsp;") == [Entity(P(1, 1), reference="&nbsp;")]

    def test_it_parses_numeric_entity(self):
        assert parse_html("&#160;") == [Entity(P(1, 1), reference="&#160;")]

    def test_it_preserves_whitespace(self):
        assert parse_html('<a\n\thref =  "foo" ></a>') == [
            OpenTag(
                P(1, 1),
                "a",
                space="\n\t",
                attrs=OrderedDict(
                    href=Attribute(
                        pos=P(2, 2),
                        name="href",
                        value="foo",
                        value_pos=P(2, 11),
                        quote='"',
                        space1=" ",
                        space2="  ",
                        space3=" ",
                    )
                ),
            ),
            CloseTag(P(2, 17), "a"),
        ]

    def test_it_parses_implicit_cdata(self):
        p = parse_html('<script>console && console.log("ni!")</script>')
        assert p == [
            OpenTag(P(1, 1), "script", space="", attrs=OrderedDict()),
            Text(P(1, 9), content='console && console.log("ni!")', cdata=True),
            CloseTag(P(1, 38), "script"),
        ]
