"""Package entrypoint for `python -m cnn_policy`."""

from __future__ import annotations

import textwrap


def main() -> None:
    print(
        textwrap.dedent(
            """
            Leorover CNN policy package

            Use one of:
              python -m cnn_policy.train --help
              python -m cnn_policy.eval --help
              python -m cnn_policy.drive --help
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
