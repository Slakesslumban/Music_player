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

from functools import wraps, partial
import sys
import threading

from markupsafe import Markup, escape_silent
from piglet.exceptions import PigletError, UndefinedError

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

Markup = Markup
escape = escape_silent
ustr = type("")
partial = partial
builtins = builtins
exc_info = sys.exc_info

data = threading.local()


if sys.version_info >= (3, 0, 0):

    def reraise(exc_info):
        raise exc_info[1].with_traceback(exc_info[2])


else:
    exec("def reraise(exc_info): raise exc_info[0], exc_info[1], exc_info[2]")


def load(relative_to, src):
    loader = relative_to.loader
    if loader is None:
        raise Exception(
            "Template has no loader attached. Ensure template "
            "was loaded via piglet.TemplateLoader.load."
        )
    template = loader.load(src, relative_to=relative_to)
    if template is relative_to:
        raise PigletError("Template {!r} may not include itself".format(template))
    return template


class Undefined(object):

    __slots__ = ("name",)

    def __init__(self, name):
        object.__setattr__(self, "name", name)

    def raise_undefined_error(self, *args, **kwargs):
        raise UndefinedError(self.name)

    __call__ = raise_undefined_error
    __hash__ = raise_undefined_error
    __sizeof__ = raise_undefined_error
    __repr__ = raise_undefined_error
    __str__ = raise_undefined_error
    __dir__ = raise_undefined_error
    __format__ = raise_undefined_error
    __subclasses__ = raise_undefined_error
    __floor__ = raise_undefined_error
    __trunc__ = raise_undefined_error
    __ceil__ = raise_undefined_error
    __cmp__ = raise_undefined_error
    __lt__ = raise_undefined_error
    __gt__ = raise_undefined_error
    __le__ = raise_undefined_error
    __ge__ = raise_undefined_error
    __eq__ = raise_undefined_error
    __ne__ = raise_undefined_error
    __getitem__ = raise_undefined_error
    __setitem__ = raise_undefined_error
    __delitem__ = raise_undefined_error
    __contains__ = raise_undefined_error
    __len__ = raise_undefined_error
    __iter__ = raise_undefined_error
    __getslice__ = raise_undefined_error
    __setslice__ = raise_undefined_error
    __reversed__ = raise_undefined_error
    __missing__ = raise_undefined_error
    __enter__ = raise_undefined_error
    __exit__ = raise_undefined_error
    __neg__ = raise_undefined_error
    __pos__ = raise_undefined_error
    __invert__ = raise_undefined_error
    __add__ = raise_undefined_error
    __sub__ = raise_undefined_error
    __mul__ = raise_undefined_error
    __div__ = raise_undefined_error
    __floordiv__ = raise_undefined_error
    __mod__ = raise_undefined_error
    __divmod__ = raise_undefined_error
    __lshift__ = raise_undefined_error
    __rshift__ = raise_undefined_error
    __and__ = raise_undefined_error
    __xor__ = raise_undefined_error
    __or__ = raise_undefined_error
    __pow__ = raise_undefined_error
    __complex__ = raise_undefined_error
    __int__ = raise_undefined_error
    __float__ = raise_undefined_error
    __index__ = raise_undefined_error
    __coerce__ = raise_undefined_error
    __get__ = raise_undefined_error
    __set__ = raise_undefined_error
    __delete__ = raise_undefined_error
    __reduce__ = raise_undefined_error
    __reduce_ex__ = raise_undefined_error
    __getinitargs__ = raise_undefined_error
    __getnewargs__ = raise_undefined_error
    __getstate__ = raise_undefined_error
    __setstate__ = raise_undefined_error
    __unicode__ = raise_undefined_error
    __long__ = raise_undefined_error
    __oct__ = raise_undefined_error
    __hex__ = raise_undefined_error
    __nonzero__ = raise_undefined_error
    __truediv_ = raise_undefined_error
    __rtruediv__ = raise_undefined_error
    __bool__ = raise_undefined_error
    __next__ = raise_undefined_error
    __getattr__ = raise_undefined_error
    __setattr__ = raise_undefined_error
    __prepare__ = raise_undefined_error
    __instancecheck__ = raise_undefined_error
    __subclasscheck__ = raise_undefined_error


class FlatOutput:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self, *args, **kwargs):
        return self.iterable

    def __str__(self):
        return "".join(self.iterable)

    __html__ = __unicode__ = __str__


def flatten(fn):
    @wraps(fn)
    def flatten(*args, **kwargs):
        return FlatOutput(fn(*args, **kwargs))

    return flatten


def munge_exception_messages(iterable, context):
    context["__piglet_exc_locations"] = []
    try:
        for item in iterable:
            yield item
    except Exception:
        cls, exc, tb = sys.exc_info()
        if getattr(exc, "__piglet_is_annotated__", False):
            raise

        saved_msg = str(exc)

        def annotated_message(self):
            locations = context["__piglet_exc_locations"]
            tb = (
                '{}"{}", line {}'.format("    " * ix, f, l)
                for ix, (f, l) in enumerate(locations)
            )

            return "{}\n" "Exception originated in code starting at {}".format(
                saved_msg, "\n".join(tb)
            )

        AnnotatedException = type(
            cls.__name__,
            (cls,),
            {
                "__str__": annotated_message,
                "__module__": cls.__module__,
                "__traceback__": tb,
                "__piglet_is_annotated__": True,
            },
        )

        try:
            exc.__class__ = AnnotatedException
        except TypeError:
            newexc = AnnotatedException.__new__(AnnotatedException)
            newexc.__init__(*exc.args)
            newexc.__dict__ = newexc.__dict__
            exc = newexc
        reraise((cls, exc, tb))
    else:
        return


@flatten
def nocontent(*args, **kwargs):
    """
    Used as default value for ``super`` argument in py:block functions.
    """
    return iter([])


def get_super(bases, current_template, name, default=nocontent):
    """
    Return the super block function from the stack of bases::

        bases = [template_a, template_b]
        get_super(bases, 'page')

    This would return a partial function application looking like this::

        partial(template_b.page, bases=[template_a])
    """

    def super(*args, **kwargs):
        try:
            eligible_bases = bases[: bases.index(current_template)]
        except ValueError:
            eligible_bases = bases[:]
        while eligible_bases:
            try:
                super_fn = getattr(eligible_bases.pop(), name)
            except AttributeError:
                continue
            return super_fn(__piglet_bases=eligible_bases, *args, **kwargs)
        return default()

    return super
