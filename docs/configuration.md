# Konfigurácia

Konfiguračné súbory sú pripravené v `configs/`, ale aktuálne sa nenačítavajú
automaticky (slúžia ako šablóna pre budúcu CLI alebo experiment runner).

## `configs/default.yaml`

### dataset
- `name`: názov datasetu (napr. `agrawal`, `elec2`, `sea`).
- `max_instances`: maximálny počet inštancií zo streamu.
- `seed`: seed pre reprodukovateľnosť.

### detector
- `type`: typ detektora (v praxi `fca`).
- `window_size`: veľkosť okna.
- `theta`: prah pre detekciu.
- `alpha`: násobok robustnej odchýlky pre adaptívny prah.

### preprocessing
- `ema_alpha`: EMA faktor pri učení kvantilov.
- `hysteresis`: v aktuálnom kóde sa nepoužíva.
- `threshold`: pevný prah pre binarizáciu (pri neadaptívnom režime).

### output
- `dir`: výstupný adresár.
- `save_csv`: ukladanie CSV logov.
- `save_plots`: ukladanie grafov.
