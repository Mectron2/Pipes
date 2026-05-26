# EasyOCR Pipe Reconstruction

Utilities for reconstructing an approximate 3D pipe model from scanned engineering drawings.

The pipeline uses:

- a side/profile drawing to extract station/elevation points;
- a plan-view drawing where the target pipe is highlighted in red;
- the extracted profile heights to generate a 3D polyline;
- the 3D polyline to export a Blender-importable OBJ tube mesh.

This is a visual reconstruction workflow, not survey-grade CAD extraction.

## Setup

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The profile parser also requires the Tesseract binary to be installed and available on `PATH`:

```bash
tesseract --version
```

On macOS, Tesseract can usually be installed with:

```bash
brew install tesseract
```

## Full Pipeline

Run all steps with one command:

```bash
.venv/bin/python pipeline.py \
  --profile-image input/pipe.png \
  --plan-image input/img.png \
  --diameter-ft 0.5 \
  --pipe-od-mm 152.4 \
  --debug-dir assets/debug-pipeline
```

If the plan drawing is not already highlighted in red, let Gemini mark the pipe route first:

```bash
.venv/bin/python pipeline.py \
  --profile-image input/pipe.png \
  --plan-image input/raw-plan.png \
  --diameter-ft 0.5 \
  --use-gemini-plan
```

Default outputs:

- `assets/points.json` - profile station/elevation points
- `assets/pipe_3d.json` - 3D pipe polyline
- `assets/pipe.obj` - Blender-importable tube mesh
- `assets/pipe_baseline_top_side.csv` - paired TOP/SIDE baseline CSV
- `assets/debug-pipeline/` - debug masks, overlays, and summary

- `input/` - source profile and plan images
- `assets/runs/run_N/points.json` - profile station/elevation points
- `assets/runs/run_N/pipe_3d.json` - 3D pipe polyline
- `assets/runs/run_N/pipe.obj` - Blender-importable tube mesh
- `assets/runs/run_N/debug-pipeline/` - debug masks, overlays, and summary
- `assets/runs/run_N/pipe_baseline_top_side.csv` - paired TOP/SIDE baseline CSV

Each CLI run uses the next numeric run folder under `assets/runs/`, for example `run_1`, `run_2`, then `run_3`.
Pass `--runs-dir /path/to/runs` to use a different base directory for auto-created `run_N` folders.
Pass `--run-dir assets/runs/run_custom` to write to a specific run directory.

Multiple side/profile drawings are supported. Pass them in route order:

```bash
.venv/bin/python pipeline.py \
  --profile-image input/profile_01.png input/profile_02.png \
  --plan-image input/img.png \
  --diameter-ft 0.5
```

The end station of one profile is treated as the start station of the next profile. The first point of each following profile is skipped to avoid duplicating the joint.

## Individual Steps

### 1. Parse Side/Profile Drawing

```bash
.venv/bin/python profile_to_points.py input/pipe.png \
  --out assets/points.json \
  --debug-dir assets/debug-profile \
  --epsilon 8
```

Multiple profiles:

```bash
.venv/bin/python profile_to_points.py input/profile_01.png input/profile_02.png \
  --out assets/points.json \
  --debug-dir assets/debug-profile
```

What it does:

- finds the profile graph area;
- reads station/elevation axis labels with OCR;
- extracts the force-main profile line;
- simplifies the line with Ramer-Douglas-Peucker;
- writes `station_ft` and `height` points.

`--epsilon` controls profile simplification:

- lower value: more profile points, closer to the source line;
- higher value: fewer profile points, smoother output.

### 2. Build 3D Polyline From Plan View

```bash
.venv/bin/python plan_to_3d.py input/img.png \
  --profile assets/points.json \
  --out assets/pipe_3d.json \
  --debug-dir assets/debug-3d \
  --sample-ft 10
```

What it does:

- segments the red pipe from the plan image;
- skeletonizes the highlighted pipe;
- orders the skeleton from the pump-station end to the far end;
- maps plan chainage to profile `station_ft`;
- interpolates profile height into `z_ft`;
- writes both pixel coordinates and approximate foot coordinates.

Important fields in `pipe_3d.json`:

```json
{
  "plan_ft_per_px": 1.10828664,
  "xy_ft_origin": {
    "x_px": 22.0,
    "y_px": 303.0
  },
  "points": [
    {
      "x_px": 22.0,
      "y_px": 303.0,
      "x_ft": 0.0,
      "y_ft": 0.0,
      "z_ft": 107.52,
      "chainage_ft": 0.09,
      "station": "0+00"
    }
  ]
}
```

The conversion is:

```text
x_ft = (x_px - origin_x_px) * plan_ft_per_px
y_ft = (origin_y_px - y_px) * plan_ft_per_px
z_ft = profile height
```

### 3. Export Blender OBJ

```bash
.venv/bin/python pipe_json_to_obj.py assets/pipe_3d.json \
  --out assets/pipe.obj \
  --diameter-ft 0.5 \
  --segments 16
```

Options:

- `--diameter-ft` - required outside diameter of the pipe in feet
- `--segments` - radial mesh detail around the pipe
- `--no-caps` - leave pipe ends open
- `--object-name` - OBJ object name

Blender can import `assets/pipe.obj` directly. The OBJ vertices are written in feet, so `1 Blender unit = 1 ft` for this generated model.

### 4. Export TOP/SIDE Baseline CSV

```bash
.venv/bin/python pipe_top_side_csv.py \
  --profile-image assets/profile-2.png \
  --plan-image assets/top-2-highlighted.png \
  --pipe-od-mm 426 \
  --out assets/pipe_baseline_top_side.csv \
  --profile-json assets/points.json \
  --pipe-3d-json assets/pipe_3d.json \
  --debug-dir assets/debug-top-side-csv
```

The CSV contains paired plan/profile rows with foot-based columns:

- `TOP` rows use reconstructed plan coordinates from the red highlighted top drawing.
- `SIDE` rows use the same stations with profile elevations from the side/profile drawing.
- Coordinates and elevations are written as `*_ft`; pipe outside diameter is written as `pipe_od_mm`.

## Accuracy Notes

The workflow is approximate.

Reliable parts:

- profile station/elevation relation, within OCR and line-extraction quality;
- `z_ft` values from the side/profile drawing;
- chainage along the reconstructed pipe path.

Approximate parts:

- plan-view XY shape comes from a red highlighted scan;
- the red line may be AI-generated or manually marked;
- the scan can have local distortion, perspective errors, or non-uniform scale;
- `x_ft/y_ft` are normalized to the profile station range, not survey-grade coordinates.

Use the output as a visual engineering reconstruction, not for construction, survey, or quantity takeoff without external validation.

## Debug Output

Profile debug directory contains masks and overlays such as:

- `01_plot_crop.png`
- `02_grid_mask.png`
- `05_selected_line.png`
- `06_overlay.png`

Plan/3D debug directory contains:

- `01_red_mask.png`
- `02_clean_mask.png`
- `03_skeleton.png`
- `04_ordered_centerline_overlay.png`

These overlays are the fastest way to verify whether the parser selected the correct line.

## Tests

Run all tests:

```bash
.venv/bin/python -m unittest \
  tests/test_profile_parser.py \
  tests/test_plan_to_3d.py \
  tests/test_pipe_top_side_csv.py \
  tests/test_pipe_json_to_obj.py \
  tests/test_pipeline.py
```

Expected result:

```text
OK
```
