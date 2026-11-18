# Configuration

Configuration templates are stored in [`configs/`](../configs/). **The current application does not automatically load these YAML files.** They serve as reference settings for experiments or future integration with the CLI and experiment runner. For supported runtime options, use the command-line interface described in the [Quick Start](quickstart.md).

## `configs/default.yaml`

### `dataset`

| Setting | Description |
| --- | --- |
| `name` | Dataset identifier, such as `agrawal`, `elec2`, or `sea`. |
| `max_instances` | Maximum number of stream observations to process. |
| `seed` | Random seed for reproducible experiments. |

### `detector`

| Setting | Description |
| --- | --- |
| `type` | Detector type; the main workflow uses `fca`. |
| `window_size` | Number of observations in the sliding window. |
| `theta` | Drift detection threshold. |
| `alpha` | Multiplier applied to the robust deviation term when calculating the adaptive detection threshold. |

### `preprocessing`

| Setting | Description |
| --- | --- |
| `ema_alpha` | Exponential moving average factor used when estimating quantiles. |
| `hysteresis` | Present in the template but not used by the current implementation. |
| `threshold` | Fixed binarization threshold for non-adaptive preprocessing. |

### `output`

| Setting | Description |
| --- | --- |
| `dir` | Output directory. |
| `save_csv` | Whether to save CSV logs. |
| `save_plots` | Whether to save plots. |

## Runtime Parameters

An example CLI invocation with explicit detector settings:

```bash
python -m fca_drift --dataset agrawal --max-instances 10000 --window-size 300 --theta 0.4 --alpha 2.0 --drift-position 5000
```

This example passes options directly to the CLI; it does **not** read `configs/default.yaml`. Parameter names and available options can be checked using:

```bash
python -m fca_drift --help
```

The detection threshold is affected by window size, preprocessing, and dataset characteristics. Values shown in examples are not universal defaults for all streams.
