# Contribution

Contributions are welcome. The code is under the GNU GPL v3 (see [LICENSE](LICENSE)).

## Development setup

1. Clone the repository and change into its folder.

2. Create a virtual environment and install the package with its development tools:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows, .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Run the tests with `pytest`, which also runs the doctests in the package:

   ```bash
   pytest
   ```

4. Lint and format with `ruff` before opening a pull request:

   ```bash
   ruff check .
   ruff format .
   ```

## Style

The code follows PEP 8 with 99-character lines, docstrings included. A docstring says what the
name and signature do not: one line for a small function, more where there is a reason, a unit
or a data quirk to record. Comments explain why, not what. Leave out `Parameters` and `Returns`
sections that only repeat the signature.

Never commit a file under `configs/`. Those files hold local data paths, and the
`fares` and `propose` configs hold an NRDP password.
