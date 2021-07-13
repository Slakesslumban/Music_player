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

from piglet.interpolate import parse_interpolations, Interpolation


def test_it_parses_text():
    assert parse_interpolations("xyzzy") == ["xyzzy"]


def test_it_handles_dollars_in_other_contexts():
    assert parse_interpolations("$11") == ["$", "11"]
    assert parse_interpolations("1$") == ["1", "$"]
    assert parse_interpolations("$$$") == ["$", "$"]
    assert parse_interpolations("$$el") == ["$", "el"]


def test_it_handles_escaped_dollars():
    assert parse_interpolations("$${") == ["$", "{"]


def test_it_detects_simple_interpolations():

    assert parse_interpolations("chips $fish") == [
        "chips ",
        Interpolation("$fish", "fish"),
    ]

    assert parse_interpolations("chips $fish[0]") == [
        "chips ",
        Interpolation("$fish[0]", "fish[0]"),
    ]

    assert parse_interpolations("chips $fish[::-1]") == [
        "chips ",
        Interpolation("$fish[::-1]", "fish[::-1]"),
    ]

    assert parse_interpolations("chips $fish[1:-1]") == [
        "chips ",
        Interpolation("$fish[1:-1]", "fish[1:-1]"),
    ]

    assert parse_interpolations("chips $eggs[0].sausages mash") == [
        "chips ",
        Interpolation("$eggs[0].sausages", "eggs[0].sausages"),
        " mash",
    ]


def test_it_detects_delimited_interpolations():
    assert parse_interpolations('fish ${{"foo": "bar"}[item]} chips') == [
        "fish ",
        Interpolation('${{"foo": "bar"}[item]}', '{"foo": "bar"}[item]'),
        " chips",
    ]
