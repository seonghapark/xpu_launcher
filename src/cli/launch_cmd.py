"""Click-based launcher command for `xpu launch`."""

from __future__ import annotations

import click

from cli.launch import build_launch_parser, run


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    }
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def launch_cmd(ctx: click.Context, args: tuple[str, ...]) -> None:
    """Launch a command across the active scheduler."""
    argv = list(args)
    if any(arg in ("-h", "--help") for arg in argv):
        parser = build_launch_parser(prog="xpu launch")
        click.echo(parser.format_help().rstrip(), color=True)
        ctx.exit(0)

    rc = run(argv)
    if rc:
        raise click.exceptions.Exit(rc)


def main() -> None:
    launch_cmd()
