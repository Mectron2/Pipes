import argparse
import heapq
import json
import math
import sys
import logging
from pathlib import Path

try:
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
except ModuleNotFoundError as exc:
    missing = exc.name
    print(
        f"Missing dependency: {missing}. Install dependencies with "
        "`pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

import profile_to_points
from logger import setup_logging


Point = tuple[float, float]
Pixel = tuple[int, int]


def red_pipe_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
    upper_red = cv2.inRange(hsv, np.array([168, 80, 80]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(lower_red, upper_red)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    if count <= 1:
        raise ValueError("Could not find a red pipe component")

    best_label = max(range(1, count), key=lambda label: stats[label, cv2.CC_STAT_AREA])
    area = int(stats[best_label, cv2.CC_STAT_AREA])
    if area < 100:
        raise ValueError("Largest red component is too small to be a pipe")

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def skeleton_pixels(mask: np.ndarray) -> list[Pixel]:
    skeleton = skeletonize(mask > 0)
    ys, xs = np.nonzero(skeleton)
    pixels = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
    if len(pixels) < 2:
        raise ValueError("Pipe skeleton has too few pixels")
    return pixels


def build_graph(pixels: list[Pixel]) -> dict[Pixel, list[tuple[Pixel, float]]]:
    pixel_set = set(pixels)
    graph: dict[Pixel, list[tuple[Pixel, float]]] = {}

    for x, y in pixels:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in pixel_set:
                    neighbors.append((neighbor, math.hypot(dx, dy)))
        graph[(x, y)] = neighbors

    return graph


def graph_endpoints(graph: dict[Pixel, list[tuple[Pixel, float]]]) -> list[Pixel]:
    endpoints = [pixel for pixel, neighbors in graph.items() if len(neighbors) == 1]
    if endpoints:
        return endpoints

    # Closed or noisy skeleton fallback: treat extreme pixels as endpoints.
    pixels = list(graph)
    left = min(pixels, key=lambda pixel: pixel[0])
    right = max(pixels, key=lambda pixel: pixel[0])
    return [left, right]


def dijkstra_path(
    graph: dict[Pixel, list[tuple[Pixel, float]]],
    start: Pixel,
) -> tuple[dict[Pixel, float], dict[Pixel, Pixel]]:
    distances = {start: 0.0}
    previous: dict[Pixel, Pixel] = {}
    queue = [(0.0, start)]

    while queue:
        distance, pixel = heapq.heappop(queue)
        if distance > distances[pixel]:
            continue

        for neighbor, weight in graph[pixel]:
            next_distance = distance + weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                previous[neighbor] = pixel
                heapq.heappush(queue, (next_distance, neighbor))

    return distances, previous


def reconstruct_path(previous: dict[Pixel, Pixel], start: Pixel, end: Pixel) -> list[Pixel]:
    path = [end]
    current = end
    while current != start:
        if current not in previous:
            raise ValueError("Could not reconstruct ordered skeleton path")
        current = previous[current]
        path.append(current)
    path.reverse()
    return path


def ordered_centerline(mask: np.ndarray) -> list[Point]:
    pixels = skeleton_pixels(mask)
    graph = build_graph(pixels)
    endpoints = graph_endpoints(graph)
    h, _ = mask.shape

    start = min(endpoints, key=lambda pixel: math.hypot(pixel[0], pixel[1] - h))
    distances, previous = dijkstra_path(graph, start)
    reachable_endpoints = [endpoint for endpoint in endpoints if endpoint in distances and endpoint != start]
    if not reachable_endpoints:
        raise ValueError("Could not find a reachable end of the pipe skeleton")

    end = max(reachable_endpoints, key=lambda pixel: distances[pixel])
    return [(float(x), float(y)) for x, y in reconstruct_path(previous, start, end)]


def cumulative_lengths(points: list[Point]) -> list[float]:
    lengths = [0.0]
    for previous, current in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(previous, current))
    return lengths


def simplify_path(points: list[Point], epsilon: float) -> list[Point]:
    if epsilon <= 0:
        return points
    return profile_to_points.rdp(points, epsilon)


def load_profile_points(profile_path: Path) -> list[dict]:
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    points = sorted(data["points"], key=lambda point: point["station_ft"])
    if len(points) < 2:
        raise ValueError("Profile JSON must contain at least two points")
    return points


def interpolate_height(profile_points: list[dict], station_ft: float) -> float:
    if station_ft <= profile_points[0]["station_ft"]:
        return float(profile_points[0]["height"])
    if station_ft >= profile_points[-1]["station_ft"]:
        return float(profile_points[-1]["height"])

    for left, right in zip(profile_points, profile_points[1:]):
        left_station = float(left["station_ft"])
        right_station = float(right["station_ft"])
        if left_station <= station_ft <= right_station:
            span = right_station - left_station
            if math.isclose(span, 0):
                return float(left["height"])
            ratio = (station_ft - left_station) / span
            return float(left["height"]) + (float(right["height"]) - float(left["height"])) * ratio

    raise ValueError(f"Could not interpolate height for station {station_ft}")


def point_at_length(points: list[Point], lengths: list[float], target_length: float) -> Point:
    if target_length <= 0:
        return points[0]
    if target_length >= lengths[-1]:
        return points[-1]

    index = int(np.searchsorted(lengths, target_length, side="right"))
    left_length = lengths[index - 1]
    right_length = lengths[index]
    left = points[index - 1]
    right = points[index]
    span = right_length - left_length
    if math.isclose(span, 0):
        return left

    ratio = (target_length - left_length) / span
    return (
        left[0] + (right[0] - left[0]) * ratio,
        left[1] + (right[1] - left[1]) * ratio,
    )


def sample_chainages(station_min: float, station_max: float, sample_ft: float) -> list[float]:
    if sample_ft <= 0:
        raise ValueError("--sample-ft must be greater than zero")

    chainages = [station_min]
    next_station = math.ceil(station_min / sample_ft) * sample_ft
    while next_station < station_max:
        if next_station > station_min:
            chainages.append(float(next_station))
        next_station += sample_ft
    chainages.append(station_max)
    return chainages


def build_3d_result(
    plan_path: Path,
    profile_path: Path,
    centerline: list[Point],
    profile_points: list[dict],
    sample_ft: float,
) -> dict:
    lengths = cumulative_lengths(centerline)
    total_px = lengths[-1]
    if math.isclose(total_px, 0):
        raise ValueError("Pipe centerline length is zero")

    station_min = float(profile_points[0]["station_ft"])
    station_max = float(profile_points[-1]["station_ft"])
    station_span = station_max - station_min
    if station_span <= 0:
        raise ValueError("Profile station range must be increasing")
    plan_ft_per_px = station_span / total_px
    origin_x, origin_y = centerline[0]

    output_points = []
    for index, chainage_ft in enumerate(sample_chainages(station_min, station_max, sample_ft)):
        target_length = (chainage_ft - station_min) / station_span * total_px
        x, y = point_at_length(centerline, lengths, target_length)
        height = interpolate_height(profile_points, chainage_ft)
        output_points.append(
            {
                "index": index,
                "x_px": round(float(x), 2),
                "y_px": round(float(y), 2),
                "x_ft": round((float(x) - origin_x) * plan_ft_per_px, 2),
                "y_ft": round((origin_y - float(y)) * plan_ft_per_px, 2),
                "z_ft": round(height, 2),
                "chainage_ft": round(float(chainage_ft), 2),
                "station": profile_to_points.format_station(chainage_ft),
                "height": round(height, 2),
            }
        )

    return {
        "source_plan": str(plan_path),
        "source_profile": str(profile_path),
        "xy_unit": "pixel",
        "xyz_unit": "ft",
        "z_unit": "ft",
        "chainage_unit": "ft",
        "total_plan_length_px": round(float(total_px), 2),
        "plan_ft_per_px": round(float(plan_ft_per_px), 8),
        "plan_px_per_ft": round(float(1 / plan_ft_per_px), 8),
        "xy_ft_origin": {
            "x_px": round(float(origin_x), 2),
            "y_px": round(float(origin_y), 2),
            "description": "x_ft/y_ft are measured from the first ordered plan centerline point; y_ft is positive upward in image coordinates.",
        },
        "station_range_ft": [round(station_min, 2), round(station_max, 2)],
        "points": output_points,
    }


def draw_centerline_overlay(image: np.ndarray, centerline: list[Point], sampled_points: list[dict]) -> np.ndarray:
    overlay = image.copy()
    path = [(int(round(x)), int(round(y))) for x, y in centerline]
    for first, second in zip(path, path[1:]):
        cv2.line(overlay, first, second, (255, 0, 0), 2)

    for point in sampled_points:
        cv2.circle(overlay, (int(round(point["x_px"])), int(round(point["y_px"]))), 2, (0, 255, 255), -1)

    if path:
        cv2.circle(overlay, path[0], 7, (0, 255, 0), -1)
        cv2.circle(overlay, path[-1], 7, (0, 0, 255), -1)
    return overlay


def write_debug(debug_dir: Path | None, image: np.ndarray, red_mask: np.ndarray, clean_mask: np.ndarray, centerline: list[Point], result: dict) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "01_red_mask.png"), red_mask)
    cv2.imwrite(str(debug_dir / "02_clean_mask.png"), clean_mask)

    skeleton_image = np.zeros(clean_mask.shape, dtype=np.uint8)
    for x, y in centerline:
        skeleton_image[int(round(y)), int(round(x))] = 255
    cv2.imwrite(str(debug_dir / "03_skeleton.png"), skeleton_image)
    cv2.imwrite(str(debug_dir / "04_ordered_centerline_overlay.png"), draw_centerline_overlay(image, centerline, result["points"]))

    meta = {
        "total_plan_length_px": result["total_plan_length_px"],
        "station_range_ft": result["station_range_ft"],
        "point_count": len(result["points"]),
        "start": result["points"][0],
        "end": result["points"][-1],
    }
    (debug_dir / "debug.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_pipe_3d(
    plan_path: Path,
    profile_path: Path,
    output_path: Path,
    debug_dir: Path | None,
    sample_ft: float,
    simplify_px: float,
) -> dict:
    image, _ = profile_to_points.load_image(plan_path)
    red_mask = red_pipe_mask(image)
    clean_mask = largest_component(red_mask)
    centerline = ordered_centerline(clean_mask)
    centerline = simplify_path(centerline, simplify_px)
    profile_points = load_profile_points(profile_path)
    result = build_3d_result(plan_path, profile_path, centerline, profile_points, sample_ft)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_debug(debug_dir, image, red_mask, clean_mask, centerline, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3D pipe polyline from a red plan-view pipe and profile heights.")
    parser.add_argument("plan_image", type=Path, help="Plan-view image with pipe highlighted in red")
    parser.add_argument("--profile", type=Path, default=Path("assets/points.json"), help="Profile JSON from profile_to_points.py")
    parser.add_argument("--out", type=Path, default=Path("assets/pipe_3d.json"), help="Output 3D JSON path")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory for debug masks and overlays")
    parser.add_argument("--sample-ft", type=float, default=10.0, help="Distance between sampled 3D points in feet")
    parser.add_argument("--simplify-px", type=float, default=2.0, help="Plan centerline simplification tolerance in pixels")
    return parser.parse_args()


def main_cli() -> int:
    setup_logging()
    args = parse_args()
    try:
        result = build_pipe_3d(args.plan_image, args.profile, args.out, args.debug_dir, args.sample_ft, args.simplify_px)
    except Exception as exc:
        logging.getLogger(__name__).exception("Error building 3D pipe JSON")
        return 1

    logging.getLogger(__name__).info("Saved %d 3D points to %s", len(result["points"]), args.out)
    if args.debug_dir:
        logging.getLogger(__name__).info("Debug artifacts saved to %s", args.debug_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
