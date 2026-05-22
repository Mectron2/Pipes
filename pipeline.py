import argparse
import json
import sys
from pathlib import Path

import main as profile_parser
import pipe_json_to_obj
import plan_to_3d


def debug_subdir(debug_dir: Path | None, name: str) -> Path | None:
    if debug_dir is None:
        return None
    return debug_dir / name


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
) -> dict:
    profile_images = profile_image if isinstance(profile_image, list) else [profile_image]
    profile = profile_parser.parse_profiles(
        profile_images,
        profile_json,
        debug_subdir(debug_dir, "profile"),
        profile_epsilon,
    )
    pipe_3d = plan_to_3d.build_pipe_3d(
        plan_image,
        profile_json,
        pipe_3d_json,
        debug_subdir(debug_dir, "plan-3d"),
        sample_ft,
        plan_simplify_px,
    )
    vertex_count, face_count = pipe_json_to_obj.convert_json_to_obj(
        pipe_3d_json,
        obj_output,
        diameter_ft,
        obj_segments,
        cap_ends,
        object_name,
    )

    summary = {
        "profile_image": [str(path) for path in profile_images],
        "plan_image": str(plan_image),
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
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


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
    return parser.parse_args()


def main_cli() -> int:
    args = parse_args()
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
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Pipeline complete")
    print(f"Profile points: {summary['profile_points']}")
    print(f"3D points: {summary['pipe_3d_points']}")
    print(f"OBJ vertices: {summary['obj_vertices']}, faces: {summary['obj_faces']}")
    print(f"OBJ saved to: {summary['outputs']['obj']}")
    if args.debug_dir:
        print(f"Debug saved to: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
