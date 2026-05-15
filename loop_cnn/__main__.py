"""Package entrypoint for `python -m loop_cnn`."""

from __future__ import annotations

import textwrap


def main() -> None:
    print(
        textwrap.dedent(
            """
            TurboPi CNN policy package

            Use one of:
              python -m loop_cnn.train --help
              python -m loop_cnn.eval --help
              python -m loop_cnn.drive --help
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
