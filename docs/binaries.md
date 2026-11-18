# Windows Executable

This guide explains how to package the command-line application as a Windows executable using PyInstaller. Building the executable is optional; the project can also be run directly from Python.

## Requirements

- Windows with Python 3.10 or later.
- Project dependencies installed from `requirements.txt`.
- Graphviz installed and available on the system for functionality that depends on it.
- PyInstaller installed in the same Python environment as the project.

## Build

Run the following commands from the project root in a Windows terminal:

```powershell
python -m pip install pyinstaller
python -m PyInstaller --onefile --name fca-drift src/fca_drift/__main__.py
```

The generated executable is located at:

```text
dist/fca-drift.exe
```

The repository also includes `fca-drift.spec`, which can be used for spec-based PyInstaller builds:

```powershell
python -m PyInstaller fca-drift.spec
```

The spec-based build may have different settings or output paths; review the spec file before building.

## Run

From the project root:

```powershell
.\dist\fca-drift.exe --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

For CLI arguments, refer to the [Quick Start](quickstart.md) or run the executable with `--help`.

## Packaging Notes

- PyInstaller builds from the active Python environment. Rebuild after changing the source code or dependencies.
- Additional files, such as content from `configs/`, can be included with PyInstaller's `--add-data` option when required.
- A one-file executable does not necessarily bundle external system tools such as Graphviz; ensure the target machine meets external runtime requirements.
- Build and test Windows executables on Windows. The instructions above describe packaging; they do not imply that a prebuilt executable is distributed in this repository.
