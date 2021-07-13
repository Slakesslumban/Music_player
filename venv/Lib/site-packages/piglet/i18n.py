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

from contextlib import contextmanager
from itertools import takewhile
import ast
import textwrap

from piglet import intermediate as im
from piglet.template import TextTemplate, HTMLTemplate
from piglet.astutil import astwalk


def get_ast_nodes(imnode):
    """
    Return an iterator over ast nodes to be searched for translation functions
    """
    if isinstance(imnode, im.InterpolateNode):
        yield ast.parse(textwrap.dedent(imnode.value))

    elif isinstance(imnode, im.WithNode):
        for target, expr in imnode.get_pairs():
            yield ast.parse(textwrap.dedent(expr))

    elif isinstance(imnode, im.InlineCodeNode):
        yield ast.parse(textwrap.dedent(imnode.pysrc))


def extract_text(fileobj, keywords, comment_tags, options, cls=TextTemplate):
    return extract(fileobj, keywords, comment_tags, options, cls)


def extract_html(fileobj, keywords, comment_tags, options, cls=HTMLTemplate):
    return extract(fileobj, keywords, comment_tags, options, cls)


def extract(fileobj, keywords, comment_tags, options, cls=HTMLTemplate):
    """
    Babel extraction function
    """
    with patch_load():
        source = fileobj.read()
        if isinstance(source, bytes):
            source = source.decode("UTF-8")

        template = cls(source)

        for _, node, _ in im.walk_tree(template.intermediate):
            for astnode in get_ast_nodes(node):
                calls = (n for n, _ in astwalk(astnode) if isinstance(n, ast.Call))
                for c in calls:
                    if isinstance(c.func, ast.Name) and c.func.id in keywords:
                        strargs = tuple(
                            x.s
                            for x in takewhile(
                                lambda x: isinstance(x, ast.Str), iter(c.args)
                            )
                        )
                        if len(strargs) == 1:
                            strargs = strargs[0]
                        if strargs:
                            yield node.pos.line, c.func.id, strargs, []

            if isinstance(node, im.TranslationNode):
                comments = [node.comment] if node.comment else []
                func_name = "_"
                yield (node.pos.line, func_name, node.get_msgstr(), comments)


@contextmanager
def patch_load():
    """
    Monkey patch piglet.runtime.load to ensure that extraction does not
    fail for templates containing <py:import> elements.
    """
    import piglet.runtime

    saved = piglet.runtime.load
    piglet.runtime.load = lambda template, *args, **kwargs: template
    yield
    piglet.runtime.load = saved
