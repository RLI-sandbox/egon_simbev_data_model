======================
egon-simbev-data-model
======================

.. container::

    .. image:: https://img.shields.io/pypi/v/egon_simbev_data_model.svg
            :target: https://pypi.python.org/pypi/egon_simbev_data_model
            :alt: PyPI Version

    .. image:: https://img.shields.io/pypi/pyversions/egon_simbev_data_model.svg
            :target: https://pypi.python.org/pypi/egon_simbev_data_model/
            :alt: PyPI Python Versions

    .. image:: https://img.shields.io/pypi/status/egon_simbev_data_model.svg
            :target: https://pypi.python.org/pypi/egon_simbev_data_model/
            :alt: PyPI Status

    .. badges from below are commendted out

    .. .. image:: https://img.shields.io/pypi/dm/egon_simbev_data_model.svg
            :target: https://pypi.python.org/pypi/egon_simbev_data_model/
            :alt: PyPI Monthly Donwloads

.. container::

    .. image:: https://img.shields.io/github/workflow/status/khelfen/egon-simbev-data-model/CI/master
            :target: https://github.com/khelfen/egon-simbev-data-model/actions/workflows/ci.yml
            :alt: CI Build Status
    .. .. image:: https://github.com/khelfen/egon-simbev-data-model/actions/workflows/ci.yml/badge.svg?branch=master

    .. image:: https://img.shields.io/github/workflow/status/khelfen/egon-simbev-data-model/Documentation/master?label=docs
            :target: https://khelfen.github.io/egon-simbev-data-model/
            :alt: Documentation Build Status
    .. .. image:: https://github.com/khelfen/egon-simbev-data-model/actions/workflows/documentation.yml/badge.svg?branch=master

    .. image:: https://img.shields.io/codecov/c/github/khelfen/egon-simbev-data-model.svg
            :target: https://codecov.io/gh/khelfen/egon-simbev-data-model
            :alt: Codecov Coverage
    .. .. image:: https://codecov.io/gh/khelfen/egon-simbev-data-model/branch/master/graph/badge.svg

    .. image:: https://img.shields.io/requires/github/khelfen/egon-simbev-data-model/master.svg
            :target: https://requires.io/github/khelfen/egon-simbev-data-model/requirements/?branch=master
            :alt: Requires.io Requirements Status
    .. .. image:: https://requires.io/github/khelfen/egon-simbev-data-model/requirements.svg?branch=master

    .. badges from below are commendted out

    .. .. image:: https://img.shields.io/travis/khelfen/egon-simbev-data-model.svg
            :target: https://travis-ci.com/khelfen/egon-simbev-data-model
            :alt: Travis CI Build Status
    .. .. image:: https://travis-ci.com/khelfen/egon-simbev-data-model.svg?branch=master

    .. .. image:: https://img.shields.io/readthedocs/egon-simbev-data-model/latest.svg
            :target: https://egon-simbev-data-model.readthedocs.io/en/latest/?badge=latest
            :alt: ReadTheDocs Documentation Build Status
    .. .. image:: https://readthedocs.org/projects/egon-simbev-data-model/badge/?version=latest

    .. .. image:: https://pyup.io/repos/github/khelfen/egon-simbev-data-model/shield.svg
            :target: https://pyup.io/repos/github/khelfen/egon-simbev-data-model/
            :alt: PyUp Updates

.. container::

    .. image:: https://img.shields.io/pypi/l/egon_simbev_data_model.svg
            :target: https://github.com/khelfen/egon-simbev-data-model/blob/master/LICENSE
            :alt: PyPI License

    .. image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2Fkhelfen%2Fegon-simbev-data-model.svg?type=shield
            :target: https://app.fossa.com/projects/git%2Bgithub.com%2Fkhelfen%2Fegon-simbev-data-model?ref=badge_shield
            :alt: FOSSA Status

.. container::

    .. image:: https://badges.gitter.im/khelfen/egon-simbev-data-model.svg
            :target: https://gitter.im/egon-simbev-data-model/community
            :alt: Gitter Chat
    .. .. image:: https://img.shields.io/gitter/room/khelfen/egon-simbev-data-model.svg

    .. image:: https://img.shields.io/badge/code%20style-black-000000.svg
            :target: https://github.com/psf/black
            :alt: Code Style: Black

eGo^n data modell for ev electromobility using simbev

* Free software: `MIT License`_
* Documentation: https://egon-simbev-data-model.readthedocs.io.

.. _`MIT License`: https://github.com/khelfen/egon-simbev-data-model/blob/master/LICENSE

Features
--------

* TODO

Install
-------

Use ``pip`` for install:

.. code-block:: console

    $ pip install -r requirements.txt

If you want to setup a development environment, use ``poetry`` instead:

.. code-block:: console

    $ # Install poetry using pipx
    $ python -m pip install pipx
    $ python -m pipx ensurepath
    $ pipx install poetry

    $ # Clone repository
    $ git clone git@github.com:RLI-sandbox/egon_simbev_data_model.git
    $ cd egon-simbev-data-model/

    $ # Install dependencies and hooks
    $ poetry install
    $ poetry run pre-commit install

Credits
-------

This package was created with Cookiecutter_ and the `elbakramer/cookiecutter-poetry`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`elbakramer/cookiecutter-poetry`: https://github.com/elbakramer/cookiecutter-poetry
