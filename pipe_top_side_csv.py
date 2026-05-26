import argparse
import csv
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import plan_to_3d
import profile_to_points


CSV_FIELDS = [
    "view",
    "polyline_id",
    "point_order",
    "point_name",
    "cad_x_ft",
    "cad_y_ft",
    "cad_z_ft",
    "station_ft",
    "plan_x_ft",
    "plan_y_ft",
    "elevation_ft",
    "pipe_od_mm",
    "segment_type",
    "note",
]


def debug_subdir(debug_dir: Path | None, name: str) -> Path | None:
    if debug_dir is None:
        return None
    return debug_dir / name


def _format_number(value: float) -> str:
    return f"{float(value):.3f}"


def _format_pipe_od(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _segment_type(points: list[dict], index: int, bend_angle_degrees: float) -> str:
    if index == 0 or index == len(points) - 1:
        return "baseline"
    previous = points[index - 1]
    current = points[index]
    following = points[index + 1]

    incoming = (
        float(current["x_ft"]) - float(previous["x_ft"]),
        float(current["y_ft"]) - float(previous["y_ft"]),
    )
    outgoing = (
        float(following["x_ft"]) - float(current["x_ft"]),
        float(following["y_ft"]) - float(current["y_ft"]),
    )
    incoming_len = math.hypot(*incoming)
    outgoing_len = math.hypot(*outgoing)
    if math.isclose(incoming_len, 0) or math.isclose(outgoing_len, 0):
        return "straight"

    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (incoming_len * outgoing_len)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    if angle >= bend_angle_degrees:
        return "bend"
    return "straight"


def _top_note(segment_type: str, index: int, point_count: int) -> str:
    if index == 0:
        return "Start point"
    if index == point_count - 1:
        return "End point"
    if segment_type == "bend":
        return "Plan direction change"
    return "Straight run"


def _csv_point_value(point: dict, key: str) -> float:
    if key not in point:
        raise ValueError(f"Pipe 3D point is missing required field: {key}")
    return float(point[key])


def build_top_side_rows(pipe_3d: dict, pipe_od_mm: float, bend_angle_degrees: float = 10.0) -> list[dict]:
    points = pipe_3d.get("points", [])
    if not points:
        raise ValueError("Pipe 3D data does not contain any points")

    pipe_od = _format_pipe_od(pipe_od_mm)
    rows = []

    for index, point in enumerate(points):
        order = index + 1
        segment_type = _segment_type(points, index, bend_angle_degrees)
        station_ft = _csv_point_value(point, "chainage_ft")
        plan_x_ft = _csv_point_value(point, "x_ft")
        plan_y_ft = _csv_point_value(point, "y_ft")
        elevation_ft = _csv_point_value(point, "z_ft")
        rows.append(
            {
                "view": "TOP",
                "polyline_id": "PIPE_BL_TOP",
                "point_order": str(order),
                "point_name": f"T{order:02d}",
                "cad_x_ft": _format_number(plan_x_ft),
                "cad_y_ft": _format_number(plan_y_ft),
                "cad_z_ft": _format_number(0.0),
                "station_ft": _format_number(station_ft),
                "plan_x_ft": _format_number(plan_x_ft),
                "plan_y_ft": _format_number(plan_y_ft),
                "elevation_ft": _format_number(elevation_ft),
                "pipe_od_mm": pipe_od,
                "segment_type": segment_type,
                "note": _top_note(segment_type, index, len(points)),
            }
        )

    for index, point in enumerate(points):
        order = index + 1
        station_ft = _csv_point_value(point, "chainage_ft")
        plan_x_ft = _csv_point_value(point, "x_ft")
        plan_y_ft = _csv_point_value(point, "y_ft")
        elevation_ft = _csv_point_value(point, "z_ft")
        rows.append(
            {
                "view": "SIDE",
                "polyline_id": "PIPE_BL_SIDE",
                "point_order": str(order),
                "point_name": f"S{order:02d}",
                "cad_x_ft": _format_number(station_ft),
                "cad_y_ft": _format_number(elevation_ft),
                "cad_z_ft": _format_number(0.0),
                "station_ft": _format_number(station_ft),
                "plan_x_ft": _format_number(plan_x_ft),
                "plan_y_ft": _format_number(plan_y_ft),
                "elevation_ft": _format_number(elevation_ft),
                "pipe_od_mm": pipe_od,
                "segment_type": "profile",
                "note": "Profile station/elevation",
            }
        )

    return rows


def write_top_side_csv(pipe_3d: dict, output_path: Path, pipe_od_mm: float, bend_angle_degrees: float = 10.0) -> list[dict]:
    rows = build_top_side_rows(pipe_3d, pipe_od_mm, bend_angle_degrees)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_top_side_csv(
    profile_images: list[Path],
    plan_image: Path,
    output_csv: Path,
    pipe_od_mm: float,
    profile_json: Path | None,
    pipe_3d_json: Path | None,
    debug_dir: Path | None,
    profile_epsilon: float,
    sample_ft: float,
    plan_simplify_px: float,
    bend_angle_degrees: float,
) -> dict:
    if not profile_images:
        raise ValueError("At least one profile image is required")

    if profile_json is not None and pipe_3d_json is not None:
        profile = profile_to_points.parse_profiles(
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
            profile_result=profile,
        )
    else:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profile_path = profile_json or tmp / "points.json"
            pipe_3d_path = pipe_3d_json or tmp / "pipe_3d.json"
            profile = profile_to_points.parse_profiles(
                profile_images,
                profile_path,
                debug_subdir(debug_dir, "profile"),
                profile_epsilon,
            )
            pipe_3d = plan_to_3d.build_pipe_3d(
                plan_image,
                profile_path,
                pipe_3d_path,
                debug_subdir(debug_dir, "plan-3d"),
                sample_ft,
                plan_simplify_px,
                profile_result=profile,
            )

    rows = write_top_side_csv(pipe_3d, output_csv, pipe_od_mm, bend_angle_degrees)

    summary = {
        "profile_image": [str(path) for path in profile_images],
        "plan_image": str(plan_image),
        "outputs": {
            "csv": str(output_csv),
            "profile_json": str(profile_json) if profile_json else None,
            "pipe_3d_json": str(pipe_3d_json) if pipe_3d_json else None,
        },
        "profile_points": len(profile["points"]),
        "pipe_3d_points": len(pipe_3d["points"]),
        "csv_rows": len(rows),
        "pipe_od_mm": pipe_od_mm,
    }

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "top_side_csv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        **summary,
        "profile": profile,
        "pipe_3d": pipe_3d,
        "csv": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse top/profile pipe drawings and export a TOP/SIDE baseline CSV.")
    parser.add_argument(
        "--profile-image",
        nargs="+",
        type=Path,
        default=[Path("assets/pipe.png")],
        help="Side/profile image(s), in route order",
    )
    parser.add_argument("--plan-image", type=Path, default=Path("assets/img.png"), help="Plan image with red pipe")
    parser.add_argument("--out", type=Path, default=Path("assets/pipe_baseline_top_side.csv"), help="Output CSV path")
    parser.add_argument("--pipe-od-mm", type=float, required=True, help="Pipe outside diameter in millimeters")
    parser.add_argument("--profile-json", type=Path, default=None, help="Optional profile JSON output path")
    parser.add_argument("--pipe-3d-json", type=Path, default=None, help="Optional 3D pipe JSON output path")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Root debug directory")
    parser.add_argument("--profile-epsilon", type=float, default=8.0, help="Profile RDP tolerance in pixels")
    parser.add_argument("--sample-ft", type=float, default=10.0, help="Distance between sampled CSV points in feet")
    parser.add_argument("--plan-simplify-px", type=float, default=2.0, help="Plan centerline simplification tolerance")
    parser.add_argument("--bend-angle-degrees", type=float, default=10.0, help="Minimum plan angle change marked as bend")
    return parser.parse_args()


def main_cli() -> int:
    args = parse_args()
    try:
        summary = run_top_side_csv(
            profile_images=args.profile_image,
            plan_image=args.plan_image,
            output_csv=args.out,
            pipe_od_mm=args.pipe_od_mm,
            profile_json=args.profile_json,
            pipe_3d_json=args.pipe_3d_json,
            debug_dir=args.debug_dir,
            profile_epsilon=args.profile_epsilon,
            sample_ft=args.sample_ft,
            plan_simplify_px=args.plan_simplify_px,
            bend_angle_degrees=args.bend_angle_degrees,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("TOP/SIDE CSV complete")
    print(f"Profile points: {summary['profile_points']}")
    print(f"3D points: {summary['pipe_3d_points']}")
    print(f"CSV rows: {summary['csv_rows']}")
    print(f"CSV saved to: {summary['outputs']['csv']}")
    if args.debug_dir:
        print(f"Debug saved to: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
