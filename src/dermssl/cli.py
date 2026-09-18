"""Shared Spanish command-line parser contracts."""

from __future__ import annotations

import argparse
import sys


class _SpanishRawDescriptionHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep multiline descriptions while localizing argparse's usage label."""

    def _format_usage(self, usage, actions, groups, prefix):
        rendered = super()._format_usage(usage, actions, groups, prefix)
        return rendered.replace("usage: ", "uso: ", 1)


def _translate_error(message: str) -> str:
    translations = (
        ("the following arguments are required:", "se requieren los siguientes argumentos:"),
        ("one of the arguments", "uno de los argumentos"),
        ("is required", "es obligatorio"),
        ("unrecognized arguments:", "argumentos no reconocidos:"),
        ("expected one argument", "se esperaba un argumento"),
        ("expected at least one argument", "se esperaba al menos un argumento"),
        ("invalid choice:", "opción no válida:"),
        ("(choose from", "(elija entre"),
    )
    for english, spanish in translations:
        message = message.replace(english, spanish)
    return message


class _SpanishArgumentParser(argparse.ArgumentParser):
    """Localize argparse's runtime validation output."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {_translate_error(message)}\n")


def build_spanish_parser(*, description: str) -> argparse.ArgumentParser:
    """Build an argparse parser whose built-in help is visible in Spanish."""
    parser = _SpanishArgumentParser(
        description=description,
        add_help=False,
        formatter_class=_SpanishRawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-h", "--help", action="help", help="muestra esta ayuda y termina"
    )
    parser._optionals.title = "opciones"
    return parser
