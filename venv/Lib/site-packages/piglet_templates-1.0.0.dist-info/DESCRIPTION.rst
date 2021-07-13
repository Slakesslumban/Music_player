Piglet Template Engine
======================

piglet-templates is a text and HTML templating engine in the genshi/kajiki
templates family.

The Piglet template engine offers:

- Template inhertitance through <py:extends>/<py:block> (similar to Jinja2)
- Compiles templates to fast python byte code.
- HTML templating: templates produce well formed HTML. Template variables are
  autoescaped to protect against XSS attacks.
- Reusable template functions.
- Good i18n support.
- Clear tracebacks from exceptions raised in templates.

`Piglet Template Engine Documentation <https://ollycope.com/software/piglet/latest/>`_
\| `Repository <https://hg.sr.ht/~olly/piglet>`_

Example piglet template:

.. code:: html

    <py:extends href="layout.html">
        <py:block name="content">
            <h1>This is the content block.</h1>
            <p>
                Hello $user.firstnames $user.lastname!
            </p>
            <p py:for="verse in poem">
                <py:for each="line in verse">$line<br/></py:for>
            </p>
        </py:block>
    </py:extends>


There's a text templating mode too:

.. code::

    Hello $user.firstnames $user.lastname!

    {% for verse in poem %}
        {% for line in verse %}$line
        {% end %}
    {% end %}


License
-------

Piglet-templates is licensed under the Apache license version 2.0.

Piglet-templates is developed by
`Olly Cope <https://ollycope.com/>`_
and was created for `skot.be <https://skot.be/>`_


