"""Top-level Click CLI for xpu."""

from __future__ import annotations

import click

from cli.__about__ import __version__
from cli.dist_cmd import dist_cmd
from cli.doctor_cmd import doctor_cmd
from cli.integrations_cmd import integrations_cmd
from cli.launch_cmd import launch_cmd
from cli.submit_cmd import submit_cmd

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def main() -> None:
    """xpu distributed utilities."""


main.add_command(launch_cmd, name="launch")
main.add_command(doctor_cmd, name="doctor")
main.add_command(submit_cmd, name="submit")
main.add_command(dist_cmd, name="dist")
main.add_command(integrations_cmd, name="integrations")
