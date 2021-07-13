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

from gettext import NullTranslations
import marshal

from piglet import compilexml
from piglet import compilett
from piglet.parse import parse_html
from piglet import compile as pigletcompile
from piglet.exceptions import PigletParseError
from piglet.runtime import data as rtdata, munge_exception_messages

ustr = type(u"")


class BaseTemplate:

    filename = None
    loader = None
    template_module = None
    codeob = None
    intermediate = None

    def __init__(
        self,
        source=None,
        filename="<string>",
        loader=None,
        translations_factory=None,
        save_pysrc_to=None,
        codeob=None,
    ):
        self.filename = filename
        self.loader = loader
        if translations_factory:
            self.translations_factory = translations_factory
        try:
            if codeob is None:
                self.intermediate = self.compile_intermediate(source)
                codeob = pigletcompile.compile_to_codeob(
                    self.intermediate, filename, saveas=save_pysrc_to
                )
            self.codeob = codeob
            self.template_module = self._get_module()
        except PigletParseError as e:
            e.filename = filename
            raise
        assert self.template_module is not None
        self.root_fn = self.template_module.__piglet_root__

    @classmethod
    def from_pysrc(
        cls,
        pysrc,
        pysrc_filename="<string>",
        filename="<string>",
        loader=None,
        translations_factory=None,
    ):

        return cls(
            filename=filename,
            codeob=compile(pysrc, pysrc_filename, "exec"),
            loader=loader,
            translations_factory=translations_factory,
        )

    def __repr__(self):
        return "<{} {!r}>".format(type(self).__name__, self.filename)

    def __call__(self, context, *args, **kwargs):
        if not hasattr(rtdata, "context"):
            rtdata.context = []
        rtdata.context.append(context)
        rtdata.exception_locations = []
        translations = self.translations_factory()
        context.update(
            {
                "_": translations.gettext,
                "gettext": translations.gettext,
                "ngettext": translations.ngettext,
            }
        )
        content = munge_exception_messages(
            self.root_fn(*args, **kwargs), context
        )
        try:
            for s in content:
                yield s
        finally:
            rtdata.context.pop()

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["template_module"]
        del d["root_fn"]
        d["codeob"] = marshal.dumps(self.codeob)
        return d

    def __setstate__(self, state):
        state["codeob"] = marshal.loads(state["codeob"])
        self.__dict__.update(state)
        self.__dict__["template_module"] = self._get_module()
        self.root_fn = self.template_module.__piglet_root__

    def _get_module(self):
        return pigletcompile.compile_to_module(
            self.codeob, self.filename, bootstrap={"__piglet_template": self}
        )

    def compile_intermediate(self):
        raise NotImplementedError()

    def render(self, context, *args, **kwargs):
        return "".join(map(ustr, self(context, *args, **kwargs)))

    def get_pysrc(self):
        """
        Return the python source for the template file
        This may return None in the case that the template was not
        compiled from source.
        """
        if self.intermediate is None:
            return None
        return pigletcompile.compile_to_source(
            self.intermediate, self.filename
        )

    def __getattr__(self, name):
        """
        Expose template functions as attributes, eg:

            >>> t = Template('<py:def function="greeting">Hello!</py:def>')
            >>> print(t.greeting)
            <function greeting at ...>
        """
        return getattr(self.template_module, name)

    def translations_factory(self):
        return NullTranslations()


class HTMLTemplate(BaseTemplate):
    def compile_intermediate(self, src):
        parsed = parse_html(src)
        return compilexml.compile_intermediate(parsed)


class TextTemplate(BaseTemplate):
    def compile_intermediate(self, src):
        parsed = compilett.parse_tt(src)
        return compilett.compile_intermediate(parsed)


Template = HTMLTemplate
