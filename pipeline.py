import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

import profile_to_points as profile_parser
import pipe_json_to_obj
import pipe_top_side_csv
import plan_to_3d
from logger import setup_logging


FEET_TO_MILLIMETERS = 304.8

DEFAULT_INPUT_DIR = Path("input")
DEFAULT_RUNS_DIR = Path("assets/runs")


def debug_subdir(debug_dir: Path | None, name: str) -> Path | None:
    if debug_dir is None:
        return None
    return debug_dir / name

def next_run_dir(runs_dir: Path) -> Path:
    max_run_number = 0
    if runs_dir.exists():
        for path in runs_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("run_"):
                continue
            suffix = path.name.removeprefix("run_")
            if suffix.isdigit():
                max_run_number = max(max_run_number, int(suffix))
    return runs_dir / f"run_{max_run_number + 1}"


def resolve_cli_paths(args: argparse.Namespace) -> argparse.Namespace:
    run_dir = args.run_dir if args.run_dir is not None else next_run_dir(args.runs_dir)
    args.run_dir = run_dir
    args.profile_json = args.profile_json if args.profile_json is not None else run_dir / "points.json"
    args.pipe_3d_json = args.pipe_3d_json if args.pipe_3d_json is not None else run_dir / "pipe_3d.json"
    args.obj = args.obj if args.obj is not None else run_dir / "pipe.obj"
    args.debug_dir = args.debug_dir if args.debug_dir is not None else run_dir / "debug-pipeline"
    return args

def prepare_plan_image(
    plan_image: Path,
    pipe_3d_json: Path,
    debug_dir: Path | None,
    use_gemini_plan: bool,
) -> tuple[Path, np.ndarray | None]:
    if not use_gemini_plan:
        return plan_image, None

    import gemini_highlighter

    gemini_output_dir = debug_subdir(debug_dir, "gemini")
    if gemini_output_dir is None:
        gemini_output_dir = pipe_3d_json.parent / "gemini"
    gemini_output_dir.mkdir(parents=True, exist_ok=True)

    highlighted_plan = gemini_output_dir / "plan.png"
    highlighted_image = gemini_highlighter.highlight_force_main_image(plan_image)
    if highlighted_image is None:
        raise RuntimeError("Gemini plan highlighting failed")
    cv2.imwrite(str(highlighted_plan), highlighted_image)
    return highlighted_plan, highlighted_image


def run_pipeline(
    profile_image: Path | list[Path],
    plan_image: Path,
    profile_json: Path,
    pipe_3d_json: Path,
    obj_output: Path,
    diameter_ft: float,
    debug_dir: Path | None,
    profile_epsilon: float,
    sample_ft: float,
    plan_simplify_px: float,
    obj_segments: int,
    cap_ends: bool,
    object_name: str,
    csv_output: Path | None = None,
    pipe_od_mm: float | None = None,
    csv_bend_angle_degrees: float = 10.0,
    use_gemini_plan: bool = False,
) -> dict:
    profile_images = profile_image if isinstance(profile_image, list) else [profile_image]
    plan_image_used, plan_image_data = prepare_plan_image(plan_image, pipe_3d_json, debug_dir, use_gemini_plan)
    profile = profile_parser.parse_profiles(
        profile_images,
        profile_json,
        debug_subdir(debug_dir, "profile"),
        profile_epsilon,
    )
    pipe_3d = plan_to_3d.build_pipe_3d(
        plan_image_used,
        profile_json,
        pipe_3d_json,
        debug_subdir(debug_dir, "plan-3d"),
        sample_ft,
        plan_simplify_px,
        profile_result=profile,
        plan_image=plan_image_data,
    )
    obj_result = pipe_json_to_obj.convert_pipe_data_to_obj(
        pipe_3d,
        obj_output,
        diameter_ft,
        obj_segments,
        cap_ends,
        object_name,
    )
    csv_rows = []
    if csv_output is not None:
        csv_pipe_od_mm = pipe_od_mm if pipe_od_mm is not None else diameter_ft * FEET_TO_MILLIMETERS
        csv_rows = pipe_top_side_csv.write_top_side_csv(
            pipe_3d,
            csv_output,
            csv_pipe_od_mm,
            csv_bend_angle_degrees,
        )

    summary = {
        "profile_image": [str(path) for path in profile_images],
        "plan_image": str(plan_image),
        "plan_image_used": str(plan_image_used),
        "gemini": {
            "plan_enabled": use_gemini_plan,
        },
        "outputs": {
            "profile_json": str(profile_json),
            "pipe_3d_json": str(pipe_3d_json),
            "obj": str(obj_output),
            "csv": str(csv_output) if csv_output else None,
        },
        "profile_points": len(profile["points"]),
        "pipe_3d_points": len(pipe_3d["points"]),
        "obj_vertices": obj_result["vertex_count"],
        "obj_faces": obj_result["face_count"],
        "diameter_ft": diameter_ft,
        "pipe_od_mm": pipe_od_mm if pipe_od_mm is not None else diameter_ft * FEET_TO_MILLIMETERS,
        "csv_rows": len(csv_rows),
    }

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        **summary,
        "profile": profile,
        "pipe_3d": pipe_3d,
        "obj": obj_result,
        "csv": csv_rows,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipe pipeline: profile image -> 3D pipe JSON -> Blender OBJ mesh."
    )
    parser.add_argument(
        "--profile-image",
        nargs="+",
        type=Path,
        default=[DEFAULT_INPUT_DIR / "pipe.png"],
        help="Side/profile image(s), in route order",
    )
    parser.add_argument("--plan-image", type=Path, default=DEFAULT_INPUT_DIR / "img.png", help="Plan image with red pipe")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Directory containing run_N output folders")
    parser.add_argument("--run-dir", type=Path, default=None, help="Specific run output directory")
    parser.add_argument("--profile-json", type=Path, default=None, help="Intermediate profile JSON")
    parser.add_argument("--pipe-3d-json", type=Path, default=None, help="Intermediate 3D JSON")
    parser.add_argument("--obj", type=Path, default=Path("assets/pipe.obj"), help="Final Blender-importable OBJ")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("assets/pipe_baseline_top_side.csv"),
        help="TOP/SIDE baseline CSV output path",
    )
    parser.add_argument("--diameter-ft", type=float, required=True, help="Pipe outside diameter in feet")
    parser.add_argument(
        "--pipe-od-mm",
        type=float,
        default=None,
        help="Pipe outside diameter in millimeters for CSV; defaults to --diameter-ft converted to mm",
    )
    parser.add_argument("--debug-dir", type=Path, default=None, help="Root debug directory")
    parser.add_argument(
        "--use-gemini-plan",
        action="store_true",
        help="Use Gemini to highlight the plan pipe in red before running OpenCV reconstruction",
    )
    parser.add_argument("--profile-epsilon", type=float, default=8.0, help="Profile RDP tolerance in pixels")
    parser.add_argument("--sample-ft", type=float, default=10.0, help="3D JSON sampling interval in feet")
    parser.add_argument("--plan-simplify-px", type=float, default=2.0, help="Plan centerline simplification tolerance")
    parser.add_argument("--csv-bend-angle-degrees", type=float, default=10.0, help="Minimum CSV plan angle change marked as bend")
    parser.add_argument("--obj-segments", type=int, default=16, help="OBJ radial segments around pipe")
    parser.add_argument("--no-caps", action="store_true", help="Leave OBJ pipe ends open")
    parser.add_argument("--object-name", default="pipe", help="OBJ object name")
    return parser.parse_args()

def main_cli() -> int:
    setup_logging()
    args = resolve_cli_paths(parse_args())
    try:
        summary = run_pipeline(
            profile_image=args.profile_image,
            plan_image=args.plan_image,
            profile_json=args.profile_json,
            pipe_3d_json=args.pipe_3d_json,
            obj_output=args.obj,
            diameter_ft=args.diameter_ft,
            debug_dir=args.debug_dir,
            profile_epsilon=args.profile_epsilon,
            sample_ft=args.sample_ft,
            plan_simplify_px=args.plan_simplify_px,
            obj_segments=args.obj_segments,
            cap_ends=not args.no_caps,
            object_name=args.object_name,
            csv_output=args.csv_output,
            pipe_od_mm=args.pipe_od_mm,
            csv_bend_angle_degrees=args.csv_bend_angle_degrees,
            use_gemini_plan=args.use_gemini_plan,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed")
        return 1

    logging.getLogger(__name__).info("Pipeline complete")
    logging.getLogger(__name__).info("Profile points: %d", summary["profile_points"])
    logging.getLogger(__name__).info("3D points: %d", summary["pipe_3d_points"])
    logging.getLogger(__name__).info("OBJ vertices: %d, faces: %d", summary["obj_vertices"], summary["obj_faces"])
    logging.getLogger(__name__).info("OBJ saved to: %s", summary["outputs"]["obj"])

    if summary["outputs"]["csv"]:
        logging.getLogger(__name__).info("CSV rows: %s", summary['csv_rows'])
        logging.getLogger(__name__).info("CSV saved to: %s", summary['outputs']['csv'])

    if args.debug_dir:
        logging.getLogger(__name__).info("Debug saved to: %s", args.debug_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
