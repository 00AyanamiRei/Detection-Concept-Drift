# Binarne vystupy (EXE)

Tato sekcia popisuje vytvorenie Windows .exe cez PyInstaller.

## Predpoklady

- Python 3.10+.
- Nainstalovane zavislosti z `requirements.txt`.
- `graphviz` nainstalovany v systeme (pre kniznicu `concepts`).

## Build (Windows)

Spusti z korena projektu:

```bash
pip install pyinstaller
pyinstaller --onefile --name fca-drift src\fca_drift\__main__.py
```

Vystupny subor bude:

- `dist\fca-drift.exe`

## Pouzitie

```bash
dist\fca-drift.exe --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

## Poznamky

- EXE sa vytvara z aktualneho Python prostredia; po zmene kodu je potrebne rebuild.
- Ak chces pribalit data (napr. `configs/`), pouzi `--add-data` v PyInstalleri.
- Pre pouzitie FCA musi byt v systeme dostupny `graphviz`.
