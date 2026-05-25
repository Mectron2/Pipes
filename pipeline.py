import argparse
import asyncio
import csv
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import profile_to_points as profile_parser
import gemini_image_edit
import pipe_json_to_obj
import plan_to_3d


logger = logging.getLogger(__name__)
RUNS_DIR = Path("assets/runs")
RUN_INPUT_DIR = RUNS_DIR / "input"
RUN_NAME_RE = re.compile(r"^run_(\d+)$")
PIPE_COORDINATE_FIELDS = [
    "index",
    "station",
    "chainage_ft",
    "x_px",
    "y_px",
    "x_ft",
    "y_ft",
    "z_ft",
    "height",
]


def debug_subdir(debug_dir: Path | None, name: str) -> Path | None:
    if debug_dir is None:
        return None
    return debug_dir / name


def default_gemini_plan_image(plan_image: Path, debug_dir: Path | None) -> Path:
    if debug_dir is not None:
        return debug_dir / "gemini" / "plan.png"
    return plan_image.with_name(f"{plan_image.stem}_gemini{plan_image.suffix}")


def default_gemini_profile_image(profile_image: Path, debug_dir: Path | None, index: int) -> Path:
    if debug_dir is not None:
        suffix = "" if index == 0 else f"_{index + 1}"
        return debug_dir / "gemini" / f"profile{suffix}{profile_image.suffix}"
    return profile_image.with_name(f"{profile_image.stem}_gemini{profile_image.suffix}")


def next_run_dir(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        match = RUN_NAME_RE.match(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return runs_dir / f"run_{max_index + 1}"


def profile_input_name(index: int, source: Path) -> str:
    if index == 0:
        return "profile.png"
    suffix = source.suffix or ".png"
    return f"profile_{index + 1:02d}{suffix}"


def copy_run_inputs(profile_images: list[Path], plan_image: Path, run_input_dir: Path) -> tuple[list[Path], Path]:
    run_input_dir.mkdir(parents=True, exist_ok=True)
    copied_profiles = []
    for index, profile_image in enumerate(profile_images):
        destination = run_input_dir / profile_input_name(index, profile_image)
        shutil.copy2(profile_image, destination)
        copied_profiles.append(destination)

    plan_destination = run_input_dir / "plan.png"
    shutil.copy2(plan_image, plan_destination)
    return copied_profiles, plan_destination


def default_gemini_profile_images(profile_images: list[Path], gemini_dir: Path) -> list[Path]:
    outputs = []
    for index, profile_image in enumerate(profile_images):
        outputs.append(gemini_dir / profile_input_name(index, profile_image))
    return outputs


def write_pipe_coordinates_csv(pipe_3d: dict, csv_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PIPE_COORDINATE_FIELDS)
        writer.writeheader()
        for point in pipe_3d["points"]:
            writer.writerow({field: point.get(field, "") for field in PIPE_COORDINATE_FIELDS})


def configure_logging(log_level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


async def edit_gemini_inputs_async(
    profile_images: list[Path],
    plan_image: Path,
    debug_dir: Path | None,
    gemini_profile_images: list[Path] | None,
    gemini_plan_image: Path | None,
    gemini_model: str,
) -> tuple[list[Path], Path]:
    async def edit_profile(index: int, source_profile_image: Path) -> Path:
        if gemini_profile_images is not None and index < len(gemini_profile_images):
            edited_profile_output = gemini_profile_images[index]
        else:
            edited_profile_output = default_gemini_profile_image(source_profile_image, debug_dir, index)

        logger.info(
            "Gemini profile edit started: source=%s output=%s",
            source_profile_image,
            edited_profile_output,
        )
        result = await gemini_image_edit.edit_profile_image_async(
            source_profile_image,
            edited_profile_output,
            gemini_model,
        )
        logger.info("Gemini profile edit finished: output=%s", result)
        return result

    async def edit_plan() -> Path:
        logger.info("Gemini plan edit started: source=%s output=%s", plan_image, edited_plan_output)
        result = await gemini_image_edit.edit_plan_image_async(plan_image, edited_plan_output, gemini_model)
        logger.info("Gemini plan edit finished: output=%s", result)
        return result

    edited_plan_output = gemini_plan_image or default_gemini_plan_image(plan_image, debug_dir)
    logger.info(
        "Gemini preprocessing started: profiles=%d plan=%s model=%s",
        len(profile_images),
        plan_image,
        gemini_model,
    )
    started = time.perf_counter()
    profile_tasks = [edit_profile(index, image) for index, image in enumerate(profile_images)]
    plan_task = edit_plan()
    *edited_profile_images, edited_plan_image = await asyncio.gather(*profile_tasks, plan_task)
    logger.info(
        "Gemini preprocessing finished: profiles=%d plan_output=%s elapsed=%.2fs",
        len(edited_profile_images),
        edited_plan_image,
        time.perf_counter() - started,
    )
    return list(edited_profile_images), edited_plan_image


async def run_pipeline_async(
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
    pipe_csv_output: Path | None = None,
    gemini_profile_images: list[Path] | None = None,
    gemini_plan_image: Path | None = None,
    gemini_model: str = gemini_image_edit.GEMINI_PLAN_MODEL,
) -> dict:
    pipeline_started = time.perf_counter()
    profile_images = profile_image if isinstance(profile_image, list) else [profile_image]
    logger.info(
        "Pipeline started: profile_images=%s plan_image=%s obj_output=%s",
        ", ".join(str(path) for path in profile_images),
        plan_image,
        obj_output,
    )

    edited_profile_images, edited_plan_image = await edit_gemini_inputs_async(
        profile_images,
        plan_image,
        debug_dir,
        gemini_profile_images,
        gemini_plan_image,
        gemini_model,
    )

    logger.info("Profile parsing started: images=%s output=%s", ", ".join(str(path) for path in edited_profile_images), profile_json)
    started = time.perf_counter()
    profile = profile_parser.parse_profiles(
        edited_profile_images,
        profile_json,
        debug_subdir(debug_dir, "profile"),
        profile_epsilon,
    )
    logger.info(
        "Profile parsing finished: points=%d output=%s elapsed=%.2fs",
        len(profile["points"]),
        profile_json,
        time.perf_counter() - started,
    )

    logger.info("3D plan reconstruction started: plan=%s profile=%s output=%s", edited_plan_image, profile_json, pipe_3d_json)
    started = time.perf_counter()
    pipe_3d = plan_to_3d.build_pipe_3d(
        edited_plan_image,
        profile_json,
        pipe_3d_json,
        debug_subdir(debug_dir, "plan-3d"),
        sample_ft,
        plan_simplify_px,
    )
    logger.info(
        "3D plan reconstruction finished: points=%d output=%s elapsed=%.2fs",
        len(pipe_3d["points"]),
        pipe_3d_json,
        time.perf_counter() - started,
    )

    logger.info("OBJ export started: input=%s output=%s diameter_ft=%s", pipe_3d_json, obj_output, diameter_ft)
    started = time.perf_counter()
    vertex_count, face_count = pipe_json_to_obj.convert_json_to_obj(
        pipe_3d_json,
        obj_output,
        diameter_ft,
        obj_segments,
        cap_ends,
        object_name,
    )
    logger.info(
        "OBJ export finished: vertices=%d faces=%d output=%s elapsed=%.2fs",
        vertex_count,
        face_count,
        obj_output,
        time.perf_counter() - started,
    )

    if pipe_csv_output is not None:
        logger.info("CSV coordinate export started: output=%s", pipe_csv_output)
        started = time.perf_counter()
        write_pipe_coordinates_csv(pipe_3d, pipe_csv_output)
        logger.info("CSV coordinate export finished: output=%s elapsed=%.2fs", pipe_csv_output, time.perf_counter() - started)

    summary = {
        "profile_image_original": [str(path) for path in profile_images],
        "profile_image_used": [str(path) for path in edited_profile_images],
        "plan_image_original": str(plan_image),
        "plan_image_used": str(edited_plan_image),
        "gemini": {
            "enabled": True,
            "model": gemini_model,
            "profile_prompt": gemini_image_edit.GEMINI_PROFILE_PROMPT.strip(),
            "plan_prompt": gemini_image_edit.GEMINI_PLAN_PROMPT.strip(),
        },
        "outputs": {
            "profile_json": str(profile_json),
            "pipe_3d_json": str(pipe_3d_json),
            "pipe_coordinates_csv": str(pipe_csv_output) if pipe_csv_output is not None else None,
            "obj": str(obj_output),
        },
        "profile_points": len(profile["points"]),
        "pipe_3d_points": len(pipe_3d["points"]),
        "obj_vertices": vertex_count,
        "obj_faces": face_count,
        "diameter_ft": diameter_ft,
    }

    if debug_dir is not None:
        logger.info("Writing pipeline summary: %s", debug_dir / "summary.json")
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Pipeline finished: elapsed=%.2fs", time.perf_counter() - pipeline_started)
    return summary


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
    pipe_csv_output: Path | None = None,
    gemini_profile_images: list[Path] | None = None,
    gemini_plan_image: Path | None = None,
    gemini_model: str = gemini_image_edit.GEMINI_PLAN_MODEL,
) -> dict:
    return asyncio.run(
        run_pipeline_async(
            profile_image=profile_image,
            plan_image=plan_image,
            profile_json=profile_json,
            pipe_3d_json=pipe_3d_json,
            obj_output=obj_output,
            diameter_ft=diameter_ft,
            debug_dir=debug_dir,
            profile_epsilon=profile_epsilon,
            sample_ft=sample_ft,
            plan_simplify_px=plan_simplify_px,
            obj_segments=obj_segments,
            cap_ends=cap_ends,
            object_name=object_name,
            pipe_csv_output=pipe_csv_output,
            gemini_profile_images=gemini_profile_images,
            gemini_plan_image=gemini_plan_image,
            gemini_model=gemini_model,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipe pipeline: profile image -> 3D pipe JSON -> Blender OBJ mesh."
    )
    parser.add_argument(
        "--profile-image",
        nargs="+",
        type=Path,
        default=None,
        help="Side/profile image(s), in route order. Defaults to assets/runs/input/profile.png.",
    )
    parser.add_argument(
        "--plan-image",
        type=Path,
        default=None,
        help="Plan image. Defaults to assets/runs/input/plan.png.",
    )
    parser.add_argument("--profile-json", type=Path, default=None, help="Intermediate profile JSON")
    parser.add_argument("--pipe-3d-json", type=Path, default=None, help="Intermediate 3D JSON")
    parser.add_argument("--pipe-coordinates-csv", type=Path, default=None, help="Output CSV with pipe coordinates")
    parser.add_argument("--obj", type=Path, default=None, help="Final Blender-importable OBJ")
    parser.add_argument("--diameter-ft", type=float, required=True, help="Pipe outside diameter in feet")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Root debug directory. Defaults to assets/runs/run_N/debug.",
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Root directory for run inputs and outputs")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to the next run_N directory.")
    parser.add_argument("--profile-epsilon", type=float, default=8.0, help="Profile RDP tolerance in pixels")
    parser.add_argument("--sample-ft", type=float, default=10.0, help="3D JSON sampling interval in feet")
    parser.add_argument("--plan-simplify-px", type=float, default=2.0, help="Plan centerline simplification tolerance")
    parser.add_argument("--obj-segments", type=int, default=16, help="OBJ radial segments around pipe")
    parser.add_argument("--no-caps", action="store_true", help="Leave OBJ pipe ends open")
    parser.add_argument("--object-name", default="pipe", help="OBJ object name")
    parser.add_argument(
        "--gemini-profile-image",
        nargs="+",
        type=Path,
        default=None,
        help="Gemini-edited profile image output(s). Defaults to assets/runs/run_N/gemini_analized/profile*.png.",
    )
    parser.add_argument(
        "--gemini-plan-image",
        type=Path,
        default=None,
        help="Gemini-edited plan image output. Defaults to assets/runs/run_N/gemini_analized/plan.png.",
    )
    parser.add_argument(
        "--gemini-model",
        default=gemini_image_edit.GEMINI_PLAN_MODEL,
        help="Gemini image editing model",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level",
    )
    return parser.parse_args()


def main_cli() -> int:
    args = parse_args()
    run_dir = args.runs_dir / args.run_name if args.run_name else next_run_dir(args.runs_dir)
    run_input_dir = run_dir / "input"
    gemini_dir = run_dir / "gemini_analized"
    output_dir = run_dir / "output"
    debug_dir = args.debug_dir or run_dir / "debug"

    configure_logging(args.log_level, debug_dir / "pipeline.log")

    source_profile_images = args.profile_image or [args.runs_dir / "input" / "profile.png"]
    source_plan_image = args.plan_image or args.runs_dir / "input" / "plan.png"

    try:
        profile_images, plan_image = copy_run_inputs(source_profile_images, source_plan_image, run_input_dir)
        gemini_profile_images = args.gemini_profile_image or default_gemini_profile_images(profile_images, gemini_dir)
        gemini_plan_image = args.gemini_plan_image or gemini_dir / "plan.png"
        profile_json = args.profile_json or output_dir / "profile_points.json"
        pipe_3d_json = args.pipe_3d_json or output_dir / "pipe_3d.json"
        pipe_coordinates_csv = args.pipe_coordinates_csv or output_dir / "pipe_coordinates.csv"
        obj_output = args.obj or output_dir / "pipe.obj"

        logger.info("Run directory prepared: %s", run_dir)
        logger.info("Run input directory: %s", run_input_dir)
        logger.info("Gemini output directory: %s", gemini_dir)
        logger.info("Output directory: %s", output_dir)
        logger.info("Debug directory: %s", debug_dir)

        summary = run_pipeline(
            profile_image=profile_images,
            plan_image=plan_image,
            profile_json=profile_json,
            pipe_3d_json=pipe_3d_json,
            obj_output=obj_output,
            diameter_ft=args.diameter_ft,
            debug_dir=debug_dir,
            profile_epsilon=args.profile_epsilon,
            sample_ft=args.sample_ft,
            plan_simplify_px=args.plan_simplify_px,
            obj_segments=args.obj_segments,
            cap_ends=not args.no_caps,
            object_name=args.object_name,
            pipe_csv_output=pipe_coordinates_csv,
            gemini_profile_images=gemini_profile_images,
            gemini_plan_image=gemini_plan_image,
            gemini_model=args.gemini_model,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Pipeline complete")
    print(f"Profile points: {summary['profile_points']}")
    print(f"3D points: {summary['pipe_3d_points']}")
    print(f"OBJ vertices: {summary['obj_vertices']}, faces: {summary['obj_faces']}")
    print(f"Gemini profile image: {', '.join(summary['profile_image_used'])}")
    print(f"Gemini plan image: {summary['plan_image_used']}")
    print(f"CSV saved to: {summary['outputs']['pipe_coordinates_csv']}")
    print(f"OBJ saved to: {summary['outputs']['obj']}")
    print(f"Run saved to: {run_dir}")
    print(f"Debug saved to: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
