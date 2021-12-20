from pathlib import Path

from dynaconf import Dynaconf

file_path = Path(__file__).parent.resolve()

settings_files = [file_path / f for f in ["settings.toml", ".secrets.toml"]]

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=settings_files,
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
