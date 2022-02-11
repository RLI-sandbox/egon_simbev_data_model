======================
egon-simbev-data-model
======================

This repository is part of the `eGo^n <https://ego-n.org/>`_ data model for EV electricity demand using SimBEV.

* Free software: `MIT License`_

.. _`MIT License`: https://github.com/RLI-sandbox/egon_simbev_data_model/blob/main/LICENSE

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
