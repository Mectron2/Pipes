import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import profile_to_points as profile_parser
import gemini_image_edit
import pipe_json_to_obj
import plan_to_3d


logger = logging.getLogger(__name__)


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
        default=[Path("assets/pipe.png")],
        help="Side/profile image(s), in route order",
    )
    parser.add_argument("--plan-image", type=Path, default=Path("assets/img.png"), help="Plan image with red pipe")
    parser.add_argument("--profile-json", type=Path, default=Path("assets/points.json"), help="Intermediate profile JSON")
    parser.add_argument("--pipe-3d-json", type=Path, default=Path("assets/pipe_3d.json"), help="Intermediate 3D JSON")
    parser.add_argument("--obj", type=Path, default=Path("assets/pipe.obj"), help="Final Blender-importable OBJ")
    parser.add_argument("--diameter-ft", type=float, required=True, help="Pipe outside diameter in feet")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Root debug directory")
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
        help="Gemini-edited profile image output(s). Defaults to debug-dir/gemini/profile*.png or profile stem + _gemini.",
    )
    parser.add_argument(
        "--gemini-plan-image",
        type=Path,
        default=None,
        help="Gemini-edited plan image output. Defaults to debug-dir/gemini/plan.png or plan stem + _gemini.",
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
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
            gemini_profile_images=args.gemini_profile_image,
            gemini_plan_image=args.gemini_plan_image,
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
    print(f"OBJ saved to: {summary['outputs']['obj']}")
    if args.debug_dir:
        print(f"Debug saved to: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
