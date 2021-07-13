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

from __future__ import unicode_literals

from os.path import basename, dirname, join as pjoin
from tempfile import NamedTemporaryFile, mkdtemp
import contextlib
import io
import os
import re
import shutil

import pytest

from piglet import TemplateLoader, TemplateNotFound, HTMLTemplate, TextTemplate
import piglet


@pytest.fixture
def tmpfile():
    with NamedTemporaryFile() as tmp:
        yield tmp


@contextlib.contextmanager
def tmpdir():
    t = mkdtemp()
    yield t
    shutil.rmtree(t)


@contextlib.contextmanager
def create_templates(paths):
    """
    Write template files to a temporary directory

    :param paths: list of (path, content) tuples
    :returns: path to temporary directory
    """
    if isinstance(paths, dict):
        paths = paths.items()

    with tmpdir() as d:
        paths = [(pjoin(d, *p.split("/")), content) for p, content in paths]
        for p, c in paths:
            try:
                os.makedirs(dirname(p))
            except OSError:
                pass
            write_template(p, c)

        try:
            yield d
        except Exception:
            import traceback

            traceback.print_exc()
            raise


def write_template(path, s):
    with io.open(path, "w", encoding="UTF-8") as f:
        f.write(s)


def test_it_loads_a_template(tmpfile):
    template_text = "<h1>whoa nelly!</h1>"
    tmpfile.write(template_text.encode("UTF-8"))
    tmpfile.flush()
    loader = TemplateLoader([dirname(tmpfile.name)])
    assert loader.load(basename(tmpfile.name)).render({}) == template_text


def test_it_caches_files(tmpfile):
    d, filename = os.path.split(tmpfile.name)
    loader = TemplateLoader([d], auto_reload=True)
    with open(tmpfile.name, "wb") as f:
        f.write(b"t1")
    t = loader.load(filename)
    assert loader.load(filename) is t
    with open(tmpfile.name, "wb") as f:
        f.write(b"t2")
    assert loader.load(filename) is not t


def test_it_auto_reloads_imported_templates():
    with tmpdir() as tmp:
        loader = TemplateLoader([tmp], auto_reload=True)
        write_template(tmp + "/fn.html", '<a py:def="foo">X</a>')
        write_template(
            tmp + "/index.html", '<py:import href="fn.html" alias="fn"/>${fn.foo()}'
        )
        t = loader.load("index.html")
        assert t.render({}) == "<a>X</a>"

        write_template(tmp + "/fn.html", '<a py:def="foo">Y</a>')
        t = loader.load("index.html")
        assert t.render({}) == "<a>Y</a>"


def test_it_ignores_changes_if_auto_reload_is_off():
    with tmpdir() as tmp:
        loader = TemplateLoader([tmp])
        write_template(tmp + "/fn.html", '<a py:def="foo">X</a>')
        write_template(
            tmp + "/index.html", '<py:import href="fn.html" alias="fn"/>${fn.foo()}'
        )
        t = loader.load("index.html")
        assert t.render({}) == "<a>X</a>"

        write_template(tmp + "/fn.html", '<a py:def="foo">Y</a>')
        t = loader.load("index.html")
        assert t.render({}) == "<a>X</a>"


def test_it_searches_path():

    template_text = "<h1>whoa nelly!</h1>"

    with tmpdir() as t1, tmpdir() as t2, tmpdir() as t3:
        loader = TemplateLoader([t1, t2, t3])
        for d in [t1, t2, t3]:
            with NamedTemporaryFile(dir=d) as f:
                f.write(template_text.encode("utf-8"))
                f.flush()
                t = loader.load(basename(f.name))
                assert t.render({}) == template_text
                assert t.filename == f.name


def test_it_loads_from_inside_template_same_dir():
    with create_templates(
        [
            ("a/t1", 't1 <py:block name="b">hello</py:block>'),
            ("a/t2", '<a py:extends="t1"><py:block name="b">t2</py:block></a>'),
        ]
    ) as d:
        loader = TemplateLoader([d])
        assert loader.load("a/t2").render({}) == "t1 t2"


def test_it_selects_class_based_on_extension():
    with create_templates(
        [("a/t1", ""), ("a/t2.txt", ""), ("a/t3.TXT", ""), ("a/t4.html", "")]
    ) as d:
        loader = TemplateLoader([d])
        assert isinstance(loader.load("a/t1"), HTMLTemplate)
        assert isinstance(loader.load("a/t2.txt"), TextTemplate)
        assert isinstance(loader.load("a/t3.TXT"), TextTemplate)
        assert isinstance(loader.load("a/t4.html"), HTMLTemplate)


def test_it_caches_python_source():
    with create_templates([("t1", "whoa nelly!")]) as d:
        loader = TemplateLoader([d], cache_dir=d)
        loader.load("t1")
        cache_dir = os.path.join(d, piglet.__version__)
        cache_file = os.path.join(
            cache_dir, next(f for f in os.listdir(cache_dir) if f.endswith(".py"))
        )
        with io.open(cache_file, "r") as f:
            cached = f.read()
        assert re.compile(r"yield u?'whoa nelly!'").search(cached) is not None

        loader._cache.clear()
        loader._level2_cache.clear()

        with io.open(cache_file, "w") as f:
            f.write(cached.replace("nelly", "smelly"))
        assert loader.load("t1").render({}) == "whoa smelly!"


def test_it_loads_from_across_template_dirs():
    templates = create_templates(
        [
            ("a/t1", 't1 <py:block name="b">hello</py:block>'),
            ("b/t2", '<a py:extends="t1"><py:block name="b">t2</py:block></a>'),
        ]
    )

    with templates as d:
        loader = TemplateLoader([pjoin(d, "a"), pjoin(d, "b")])
        assert loader.load("t2").render({}) == "t1 t2"

        TemplateLoader([pjoin(d, "b"), pjoin(d, "a")])
        assert loader.load("t2").render({}) == "t1 t2"


def test_it_loads_relative():
    templates = create_templates(
        [
            ("a/t1", 'a-t1 <py:block name="b">hello</py:block>'),
            ("b/t1", 'b-t1 <py:block name="b">hello</py:block>'),
            ("b/t2", '<a py:extends="./t1"><py:block name="b">t2</py:block></a>'),
        ]
    )
    with templates as d:
        loader = TemplateLoader([pjoin(d, "a"), pjoin(d, "b")])
        assert loader.load("t2").render({}) == "b-t1 t2"


def test_it_loads_relative_parent_directory():
    templates = create_templates(
        [
            ("layout", 'layout <py:block name="B">hello</py:block>'),
            ("a/layout", 'a/layout <py:block name="B">hello</py:block>'),
            ("a/b/layout", 'a/b/layout <py:block name="B">hello</py:block>'),
            (
                "a/b/c",
                '<a py:extends="../layout"><py:block name="B">a/b/c</py:block></a>',
            ),
        ]
    )
    with templates as d:
        loader = TemplateLoader([pjoin(d, "a")])
        assert loader.load("b/c").render({}) == "a/layout a/b/c"


def test_it_does_not_break_out_of_search_path():
    templates = create_templates(
        [
            ("layout", "OUTSIDE OF SEARCH PATH"),
            ("tpl_root/layout", "layout"),
            ("tpl_root/b", '<a py:extends="../layout"></a>'),
        ]
    )
    with templates as d:
        loader = TemplateLoader([pjoin(d, "tpl_root")])
        template = loader.load("b")
        with pytest.raises(TemplateNotFound):
            template.render({})


def test_it_loads_in_search_path_order():
    templates = create_templates(
        [
            ("a/t1", 'a-t1 <py:block name="b">hello</py:block>'),
            ("b/t1", 'b-t1 <py:block name="b">hello</py:block>'),
            ("b/t2", '<a py:extends="t1"><py:block name="b">t2</py:block></a>'),
        ]
    )
    with templates as d:
        loader = TemplateLoader([pjoin(d, "a"), pjoin(d, "b")])
        assert loader.load("t2").render({}) == "a-t1 t2"


def test_it_loads_absolute_paths():
    templates = create_templates([("a/t1", "this is a/t1"), ("b/t1", "this is b/t1")])
    with templates as d:
        loader = TemplateLoader([pjoin(d, "a")])
        assert loader.load(pjoin(d, "a/t1")).render({}) == "this is a/t1"
        with pytest.raises(TemplateNotFound):
            loader.load(pjoin(d, "b/t1"))
        loader.allow_absolute_paths = True
        assert loader.load(pjoin(d, "b/t1")).render({}) == "this is b/t1"


def test_it_doesnt_load_current_template():
    templates = create_templates(
        [
            ("a/page", 'a-page <py:extends href="layout"></py:extends>'),
            ("a/layout", 'a-layout <py:extends href="layout"></py:extends>'),
            ("b/layout", "b-layout"),
        ]
    )
    with templates as d:
        loader = TemplateLoader([pjoin(d, "a"), pjoin(d, "b")])
        assert loader.load("page").render({}) == "a-page a-layout b-layout"
