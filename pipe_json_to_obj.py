import argparse
import json
import math
import sys
import logging
from pathlib import Path
from logger import setup_logging


Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    size = length(a)
    if math.isclose(size, 0):
        raise ValueError("Cannot normalize zero-length vector")
    return (a[0] / size, a[1] / size, a[2] / size)


def project_perpendicular(vector: Vec3, normal: Vec3) -> Vec3:
    return sub(vector, mul(normal, dot(vector, normal)))


def load_pipe_points(path: Path) -> list[Vec3]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = []
    for item in data.get("points", []):
        try:
            points.append((float(item["x_ft"]), float(item["y_ft"]), float(item["z_ft"])))
        except KeyError as exc:
            raise ValueError("Input JSON must contain points with x_ft, y_ft, and z_ft") from exc

    if len(points) < 2:
        raise ValueError("Input JSON must contain at least two 3D points")
    return remove_duplicate_points(points)


def remove_duplicate_points(points: list[Vec3]) -> list[Vec3]:
    deduped = [points[0]]
    for point in points[1:]:
        if length(sub(point, deduped[-1])) > 1e-8:
            deduped.append(point)
    if len(deduped) < 2:
        raise ValueError("Input polyline collapses to fewer than two unique points")
    return deduped


def tangent_at(points: list[Vec3], index: int) -> Vec3:
    if index == 0:
        return normalize(sub(points[1], points[0]))
    if index == len(points) - 1:
        return normalize(sub(points[-1], points[-2]))
    return normalize(sub(points[index + 1], points[index - 1]))


def initial_frame(tangent: Vec3) -> tuple[Vec3, Vec3]:
    up = (0.0, 0.0, 1.0)
    if abs(dot(tangent, up)) > 0.95:
        up = (0.0, 1.0, 0.0)

    normal = normalize(project_perpendicular(up, tangent))
    binormal = normalize(cross(tangent, normal))
    return normal, binormal


def transport_frame(previous_normal: Vec3, tangent: Vec3) -> tuple[Vec3, Vec3]:
    normal = project_perpendicular(previous_normal, tangent)
    if length(normal) < 1e-8:
        normal, binormal = initial_frame(tangent)
        return normal, binormal

    normal = normalize(normal)
    binormal = normalize(cross(tangent, normal))
    return normal, binormal


def build_tube_mesh(
    points: list[Vec3],
    diameter: float,
    radial_segments: int,
    cap_ends: bool,
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    if diameter <= 0:
        raise ValueError("Pipe diameter must be greater than zero")
    if radial_segments < 3:
        raise ValueError("Radial segments must be at least 3")

    radius = diameter / 2
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []

    tangent = tangent_at(points, 0)
    normal, binormal = initial_frame(tangent)

    for index, center in enumerate(points):
        tangent = tangent_at(points, index)
        if index > 0:
            normal, binormal = transport_frame(normal, tangent)

        for segment in range(radial_segments):
            angle = 2 * math.pi * segment / radial_segments
            offset = add(mul(normal, math.cos(angle) * radius), mul(binormal, math.sin(angle) * radius))
            vertices.append(add(center, offset))

    for ring in range(len(points) - 1):
        current = ring * radial_segments
        nxt = (ring + 1) * radial_segments
        for segment in range(radial_segments):
            a = current + segment + 1
            b = current + ((segment + 1) % radial_segments) + 1
            c = nxt + ((segment + 1) % radial_segments) + 1
            d = nxt + segment + 1
            faces.append((a, b, c, d))

    if cap_ends:
        start_center = len(vertices) + 1
        vertices.append(points[0])
        end_center = len(vertices) + 1
        vertices.append(points[-1])

        start_ring = 0
        end_ring = (len(points) - 1) * radial_segments
        for segment in range(radial_segments):
            a = start_ring + segment + 1
            b = start_ring + ((segment + 1) % radial_segments) + 1
            faces.append((start_center, b, a))

            c = end_ring + segment + 1
            d = end_ring + ((segment + 1) % radial_segments) + 1
            faces.append((end_center, c, d))

    return vertices, faces


def write_obj(path: Path, vertices: list[Vec3], faces: list[tuple[int, ...]], object_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# Pipe mesh generated from pipe_3d.json\n")
        file.write("# Units: feet\n")
        file.write(f"o {object_name}\n")
        for vertex in vertices:
            file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            file.write("f " + " ".join(str(index) for index in face) + "\n")


def convert_json_to_obj(
    input_path: Path,
    output_path: Path,
    diameter_ft: float,
    radial_segments: int,
    cap_ends: bool,
    object_name: str,
) -> tuple[int, int]:
    points = load_pipe_points(input_path)
    vertices, faces = build_tube_mesh(points, diameter_ft, radial_segments, cap_ends)
    write_obj(output_path, vertices, faces, object_name)
    return len(vertices), len(faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pipe_3d.json into a Blender-importable OBJ tube mesh.")
    parser.add_argument("input", type=Path, help="Input pipe_3d.json with x_ft/y_ft/z_ft points")
    parser.add_argument("--out", type=Path, default=Path("assets/pipe.obj"), help="Output OBJ path")
    parser.add_argument("--diameter-ft", type=float, required=True, help="Pipe outside diameter in feet")
    parser.add_argument("--segments", type=int, default=16, help="Radial mesh segments around the pipe")
    parser.add_argument("--no-caps", action="store_true", help="Leave pipe ends open")
    parser.add_argument("--object-name", default="pipe", help="OBJ object name")
    return parser.parse_args()


def main_cli() -> int:
    setup_logging()
    args = parse_args()
    try:
        vertex_count, face_count = convert_json_to_obj(
            args.input,
            args.out,
            args.diameter_ft,
            args.segments,
            not args.no_caps,
            args.object_name,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Error converting JSON to OBJ")
        return 1

    logging.getLogger(__name__).info("Saved OBJ to %s", args.out)
    logging.getLogger(__name__).info("Vertices: %d, faces: %d", vertex_count, face_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
