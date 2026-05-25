import argparse
import heapq
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pytesseract
except ModuleNotFoundError as exc:
    missing = exc.name
    print(
        f"Missing dependency: {missing}. Install dependencies with "
        "`pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


STATION_RE = re.compile(r"^(\d+)\+(\d{1,2})$")
NUMBER_RE = re.compile(r"^\d{2,3}$")
TEXT_REMOVAL_MIN_CONFIDENCE = 80.0


@dataclass(frozen=True)
class Token:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def cx(self) -> float:
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        return self.top + self.height / 2


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float
    labels_used: list

    def value_at(self, pixel: float) -> float:
        return self.slope * pixel + self.intercept


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255)
        image = (rgb * alpha[:, :, None] + white * (1 - alpha[:, :, None])).astype(np.uint8)

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[:, :, :3]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def enhance_profile_contrast(bgr: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr, enhanced_gray


def ensure_dir(path: Path | None) -> None:
    if path:
        path.mkdir(parents=True, exist_ok=True)


def write_debug(debug_dir: Path | None, name: str, image: np.ndarray) -> None:
    if not debug_dir:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(str(debug_dir / name), image)


def dark_mask(gray: np.ndarray, threshold: int = 215) -> np.ndarray:
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def strong_mask(gray: np.ndarray, threshold: int = 170) -> np.ndarray:
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def grouped_positions(projection: np.ndarray, minimum: float) -> list[int]:
    positions: list[int] = []
    in_run = False
    start = 0

    for i, value in enumerate(projection):
        if value >= minimum and not in_run:
            in_run = True
            start = i
        elif value < minimum and in_run:
            in_run = False
            positions.append((start + i - 1) // 2)

    if in_run:
        positions.append((start + len(projection) - 1) // 2)

    return positions


def find_plot_bounds(gray: np.ndarray) -> tuple[int, int, int, int]:
    mask = dark_mask(gray)
    h, w = mask.shape

    row_hits = np.count_nonzero(mask, axis=1)
    col_hits = np.count_nonzero(mask, axis=0)
    rows = grouped_positions(row_hits, minimum=w * 0.45)
    cols = grouped_positions(col_hits, minimum=h * 0.45)

    if len(rows) >= 2 and len(cols) >= 2:
        y0, y1 = min(rows), max(rows)
        x0, x1 = min(cols), max(cols)
        if (x1 - x0) > w * 0.5 and (y1 - y0) > h * 0.5:
            return refine_plot_bounds_with_grid(gray, (x0, y0, x1 - x0 + 1, y1 - y0 + 1))

    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("No dark pixels found in image")
    x, y, bw, bh = cv2.boundingRect(coords)
    return refine_plot_bounds_with_grid(gray, (x, y, bw, bh))


def _merge_positions(positions: list[int], max_gap: int = 3) -> list[int]:
    if not positions:
        return []

    positions = sorted(positions)
    groups = [[positions[0]]]

    for p in positions[1:]:
        if p - groups[-1][-1] <= max_gap:
            groups[-1].append(p)
        else:
            groups.append([p])

    return [int(round(np.mean(g))) for g in groups]


def _projection_grid_positions(
    mask: np.ndarray,
    *,
    axis: int,
    close_kernel: tuple[int, int],
    minimum_ratio: float,
    merge_gap_px: int,
) -> list[int]:
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel),
    )

    if axis == 1:
        projection = np.count_nonzero(closed, axis=1)
        minimum = closed.shape[1] * minimum_ratio
    else:
        projection = np.count_nonzero(closed, axis=0)
        minimum = closed.shape[0] * minimum_ratio

    return _merge_positions(grouped_positions(projection, minimum), max_gap=merge_gap_px)


def _dominant_grid_spacing(positions: list[int], limit: int) -> int | None:
    if len(positions) < 4:
        return None

    buckets: dict[int, int] = {}
    max_spacing = max(40, int(limit * 0.35))
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            diff = second - first
            if 35 <= diff <= max_spacing:
                bucket = int(round(diff / 5) * 5)
                buckets[bucket] = buckets.get(bucket, 0) + 1

    if not buckets:
        return None

    return min(
        buckets,
        key=lambda bucket: (-buckets[bucket], bucket),
    )


def _regularize_grid_positions(positions: list[int], limit: int, merge_gap_px: int) -> list[int]:
    positions = _merge_positions(positions, max_gap=merge_gap_px)
    spacing = _dominant_grid_spacing(positions, limit)
    if spacing is None:
        return positions

    tolerance = max(6, int(round(spacing * 0.08)))
    best_start = None
    best_matches: list[tuple[int, int]] = []

    for anchor in positions:
        start = anchor
        while start - spacing >= 0:
            start -= spacing

        matches = []
        for position in positions:
            index = int(round((position - start) / spacing))
            expected = start + index * spacing
            if abs(position - expected) <= tolerance:
                matches.append((position, index))

        if len(matches) > len(best_matches):
            best_start = start
            best_matches = matches

    if best_start is None or len(best_matches) < max(4, int(len(positions) * 0.6)):
        return positions

    matched_indices = [index for _, index in best_matches]
    completed = []
    for index in range(min(matched_indices), max(matched_indices) + 1):
        expected = best_start + index * spacing
        nearest = min(positions, key=lambda position: abs(position - expected))
        if abs(nearest - expected) <= tolerance:
            expected = nearest
        if 0 <= expected < limit:
            completed.append(int(expected))

    return _merge_positions(completed, max_gap=merge_gap_px)


def refine_plot_bounds_with_grid(
    gray: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x0, y0, width, height = bounds
    crop = gray[y0 : y0 + height, x0 : x0 + width]
    if crop.size == 0:
        return bounds

    _, rows, cols = detect_grid(crop)
    if len(rows) < 4 or len(cols) < 4:
        return bounds

    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    refined_width = right - left + 1
    refined_height = bottom - top + 1
    if refined_width < width * 0.5 or refined_height < height * 0.5:
        return bounds

    return x0 + left, y0 + top, refined_width, refined_height


def detect_grid(
    crop_gray: np.ndarray,
    *,
    min_line_ratio: float = 0.25,
    max_offset_px: int = 2,
    merge_gap_px: int = 4,
) -> tuple[np.ndarray, list[int], list[int]]:
    mask = dark_mask(crop_gray)
    h, w = mask.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, w // 28), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(40, h // 18))
    )

    horizontal_candidate = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_candidate = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    min_horizontal_len = int(w * min_line_ratio)
    min_vertical_len = int(h * min_line_ratio)

    horizontal_lines = cv2.HoughLinesP(
        horizontal_candidate,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, w // 20),
        minLineLength=min_horizontal_len,
        maxLineGap=max(5, w // 80),
    )

    vertical_lines = cv2.HoughLinesP(
        vertical_candidate,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, h // 20),
        minLineLength=min_vertical_len,
        maxLineGap=max(5, h // 80),
    )

    grid = np.zeros_like(mask)

    row_positions: list[int] = []
    col_positions: list[int] = []

    if horizontal_lines is not None:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]

            if abs(y1 - y2) > max_offset_px:
                continue

            length = abs(x2 - x1)
            if length < min_horizontal_len:
                continue

            y = int(round((y1 + y2) / 2))
            row_positions.append(y)

            cv2.line(
                grid,
                (min(x1, x2), y),
                (max(x1, x2), y),
                255,
                1
            )

    if vertical_lines is not None:
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]

            if abs(x1 - x2) > max_offset_px:
                continue

            length = abs(y2 - y1)
            if length < min_vertical_len:
                continue

            x = int(round((x1 + x2) / 2))
            col_positions.append(x)

            cv2.line(
                grid,
                (x, min(y1, y2)),
                (x, max(y1, y2)),
                255,
                1
            )

    projection_merge_gap = max(merge_gap_px, 8)
    row_positions.extend(
        _projection_grid_positions(
            mask,
            axis=1,
            close_kernel=(max(15, w // 120), 1),
            minimum_ratio=0.18,
            merge_gap_px=projection_merge_gap,
        )
    )
    col_positions.extend(
        _projection_grid_positions(
            mask,
            axis=0,
            close_kernel=(1, max(15, h // 120)),
            minimum_ratio=0.18,
            merge_gap_px=projection_merge_gap,
        )
    )

    rows = _regularize_grid_positions(row_positions, h, projection_merge_gap)
    cols = _regularize_grid_positions(col_positions, w, projection_merge_gap)

    if rows and cols:
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        for y in rows:
            cv2.line(grid, (left, y), (right, y), 255, 1)
        for x in cols:
            cv2.line(grid, (x, top), (x, bottom), 255, 1)

    return grid, rows, cols


def ocr_tokens(image: np.ndarray) -> list[Token]:
    config = "--psm 11 -c tessedit_char_whitelist=0123456789+"
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    tokens: list[Token] = []
    for i, raw_text in enumerate(data["text"]):
        text = re.sub(r"[^0-9+]", "", raw_text.strip())
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except ValueError:
            confidence = -1

        if confidence < 0:
            continue

        tokens.append(
            Token(
                text=text,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=confidence,
            )
        )
    return tokens

def get_ocr_tokens(image: np.ndarray, min_confidence: float = 0.0) -> list[Token]:
    config = "--psm 11"
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    tokens: list[Token] = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except ValueError:
            confidence = -1

        if confidence < min_confidence:
            continue

        tokens.append(
            Token(
                text=text,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=confidence,
            )
        )
    return tokens

def station_to_feet(text: str) -> float | None:
    match = STATION_RE.match(text)
    if not match:
        return None
    hundreds = int(match.group(1))
    offset = int(match.group(2))
    return hundreds * 100 + offset


def format_station(feet: float) -> str:
    rounded = int(round(feet))
    hundreds, offset = divmod(max(0, rounded), 100)
    return f"{hundreds}+{offset:02d}"


def interpolated_station_labels(labels: list[str], step: int = 100) -> list[str]:
    stations = [station_to_feet(label) for label in labels]
    stations = [station for station in stations if station is not None]
    if len(stations) < 2:
        return labels

    start = int(math.floor(min(stations) / step) * step)
    end = int(math.ceil(max(stations) / step) * step)
    return [format_station(station) for station in range(start, end + step, step)]


def linear_calibration(samples: list[tuple[float, float, str]]) -> Calibration:
    if len(samples) < 2:
        raise ValueError("At least two samples are required for calibration")
    pixels = np.array([sample[0] for sample in samples], dtype=np.float64)
    values = np.array([sample[1] for sample in samples], dtype=np.float64)
    slope, intercept = np.polyfit(pixels, values, 1)
    labels = [sample[2] for sample in sorted(samples, key=lambda sample: sample[1])]
    return Calibration(float(slope), float(intercept), labels)


def robust_linear_calibration(
    samples: list[tuple[float, float, str]],
    residual_tolerance: float,
) -> Calibration:
    if len(samples) < 2:
        raise ValueError("At least two samples are required for calibration")

    best_inliers: list[tuple[float, float, str]] = []
    best_error = float("inf")

    for i, first in enumerate(samples):
        for second in samples[i + 1 :]:
            x1, y1, _ = first
            x2, y2, _ = second
            if math.isclose(x1, x2) or math.isclose(y1, y2):
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            inliers = [
                sample
                for sample in samples
                if abs((slope * sample[0] + intercept) - sample[1]) <= residual_tolerance
            ]
            if len(inliers) < 2:
                continue

            errors = [abs((slope * sample[0] + intercept) - sample[1]) for sample in inliers]
            median_error = float(np.median(errors))
            better = len(inliers) > len(best_inliers)
            same_count_better = len(inliers) == len(best_inliers) and median_error < best_error
            if better or same_count_better:
                best_inliers = inliers
                best_error = median_error

    if len(best_inliers) < 2:
        return linear_calibration(samples)

    return linear_calibration(best_inliers)


def calibrate_x(tokens: list[Token], bounds: tuple[int, int, int, int]) -> Calibration:
    x0, y0, w, h = bounds
    samples: list[tuple[float, float, str]] = []

    for token in tokens:
        station = station_to_feet(token.text)
        if station is None:
            continue
        near_bottom = token.cy >= y0 + h - 45
        within_x = x0 - 25 <= token.cx <= x0 + w + 25
        if near_bottom and within_x:
            samples.append((token.cx, station, token.text))

    if len(samples) < 2:
        raise ValueError("Could not calibrate X axis: fewer than two station labels found")
    calibration = robust_linear_calibration(samples, residual_tolerance=35.0)
    return Calibration(
        calibration.slope,
        calibration.intercept,
        interpolated_station_labels(calibration.labels_used),
    )


def calibrate_y(tokens: list[Token], bounds: tuple[int, int, int, int]) -> Calibration:
    x0, y0, w, h = bounds
    samples: list[tuple[float, float, str]] = []

    for token in tokens:
        if not NUMBER_RE.match(token.text):
            continue
        value = int(token.text)
        if not 50 <= value <= 250:
            continue
        near_axis = token.cx <= x0 + 75 or token.cx >= x0 + w - 75
        within_y = y0 - 25 <= token.cy <= y0 + h + 25
        if near_axis and within_y:
            samples.append((token.cy, float(value), token.text))

    samples = dedupe_samples(samples)
    if len(samples) < 2:
        raise ValueError("Could not calibrate Y axis: fewer than two height labels found")
    return robust_linear_calibration(samples, residual_tolerance=1.5)


def dedupe_samples(samples: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    samples = sorted(samples, key=lambda item: (item[1], item[0]))
    deduped: list[tuple[float, float, str]] = []
    for pixel, value, label in samples:
        if deduped and math.isclose(deduped[-1][1], value, abs_tol=0.01):
            previous = deduped[-1]
            # Keep the more axis-like sample when both sides have the same label.
            if abs(pixel) < abs(previous[0]):
                deduped[-1] = (pixel, value, label)
            continue
        deduped.append((pixel, value, label))
    return deduped


def token_mask(shape: tuple[int, int], tokens: list[Token], bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = bounds
    mask = np.zeros(shape, dtype=np.uint8)
    for token in tokens:
        left = max(0, token.left - x0 - 8)
        top = max(0, token.top - y0 - 5)
        right = min(w - 1, token.left + token.width - x0 + 8)
        bottom = min(h - 1, token.top + token.height - y0 + 5)
        if right > left and bottom > top:
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    return mask


def extract_line_mask(
    crop_gray: np.ndarray,
    grid: np.ndarray,
    tokens: list[Token],
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = crop_gray.shape
    line = strong_mask(crop_gray)

    grid_dilated = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    line = cv2.bitwise_and(line, cv2.bitwise_not(grid_dilated))
    line = cv2.bitwise_and(line, cv2.bitwise_not(token_mask((h, w), tokens, bounds)))

    line = cv2.morphologyEx(line, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    closed = cv2.morphologyEx(line, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5)))
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return line, closed


WHITE_THRESHOLD = 127


def is_white(image: np.ndarray, x: int, y: int) -> bool:
    return image[y, x] > WHITE_THRESHOLD


def resize_image(image: np.ndarray, scale: float = 0.25) -> np.ndarray:
    if not 0 < scale <= 1:
        raise ValueError("scale must be in [0, 1]")

    if scale == 1:
        return image.copy()

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    return resized


def get_neighbors(x: int, y: int, width: int, height: int):
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    for dx, dy in directions:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def step_cost(image: np.ndarray, current, neighbor):
    x1, y1 = current
    x2, y2 = neighbor

    current_white = is_white(image, x1, y1)
    neighbor_white = is_white(image, x2, y2)

    cost = 1

    if not neighbor_white:
        cost += 50

    if current_white and not neighbor_white:
        cost += 200

    return cost


def add_cost(a, b):
    return a + b


def find_left_right_path(image: np.ndarray):
    height, width = image.shape[:2]

    left_margin = max(1, int(width * 0.02))
    right_margin = int(width * 0.98)

    starts = []
    goals = set()

    for y in range(height):
        for x in range(left_margin):
            if is_white(image, x, y):
                starts.append((int(x), int(y)))

    for y in range(height):
        for x in range(right_margin, width):
            if is_white(image, x, y):
                goals.add((int(x), int(y)))

    if not starts:
        return None, None

    if not goals:
        return None, None

    dist = {}
    parent = {}
    queue = []

    for start in starts:
        dist[start] = 0
        parent[start] = None
        heapq.heappush(queue, (0, start))

    final_goal = None

    while queue:
        current_cost, current = heapq.heappop(queue)

        if current_cost != dist.get(current, math.inf):
            continue

        if current in goals:
            final_goal = current
            break

        x, y = current

        for neighbor in get_neighbors(x, y, width, height):
            new_cost = add_cost(
                current_cost,
                step_cost(image, current, neighbor)
            )

            if new_cost < dist.get(neighbor, math.inf):
                dist[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(queue, (new_cost, neighbor))

    if final_goal is None:
        return None, None

    path = []
    node = final_goal

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()

    return path, dist[final_goal]


def scale_path_to_original(path, scale: float):
    if path is None:
        return None

    original_path = []

    for x, y in path:
        original_x = float(x / scale)
        original_y = float(y / scale)
        original_path.append((original_x, original_y))

    return original_path


def trace_centerline(line: np.ndarray, max_gap: int = 100) -> list[tuple[float, float]]:
    scale = 0.25
    h, w = line.shape
    
    small_image = resize_image(line, scale=scale)
    
    path_small, _ = find_left_right_path(small_image)
    if path_small is None:
        raise ValueError("Could not find a path from left to right")
        
    path_original = scale_path_to_original(path_small, scale)
    if path_original is None or not path_original:
        raise ValueError("Original path is empty")
        
    monotonic_path = []
    last_x = -1
    for px, py in path_original:
        if px > last_x:
            monotonic_path.append((px, py))
            last_x = px
            
    if not monotonic_path:
        raise ValueError("Path is empty after monotonic filter")

    interpolated: list[tuple[float, float]] = []
    for current, nxt in zip(monotonic_path, monotonic_path[1:]):
        interpolated.append(current)
        gap = int(nxt[0] - current[0])
        if 1 < gap <= max_gap:
            for offset in range(1, gap):
                ratio = offset / gap
                y = current[1] + (nxt[1] - current[1]) * ratio
                interpolated.append((current[0] + offset, y))
    interpolated.append(monotonic_path[-1])

    return median_smooth(interpolated, window=25, height=h)


def median_smooth(points: list[tuple[float, float]], window: int, height: int) -> list[tuple[float, float]]:
    radius = window // 2
    ys = np.array([point[1] for point in points], dtype=np.float64)
    smoothed: list[tuple[float, float]] = []

    for i, (x, _) in enumerate(points):
        start = max(0, i - radius)
        end = min(len(points), i + radius + 1)
        y = float(np.median(ys[start:end]))
        smoothed.append((x, min(max(y, 0.0), float(height - 1))))

    return smoothed


def perpendicular_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        return math.dist(point, start)
    px, py = point
    sx, sy = start
    ex, ey = end
    numerator = abs((ey - sy) * px - (ex - sx) * py + ex * sy - ey * sx)
    denominator = math.hypot(ey - sy, ex - sx)
    return numerator / denominator


def rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]
    max_distance = 0.0
    max_index = 0

    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], start, end)
        if distance > max_distance:
            max_distance = distance
            max_index = i

    if max_distance > epsilon:
        left = rdp(points[: max_index + 1], epsilon)
        right = rdp(points[max_index:], epsilon)
        return left[:-1] + right

    return [start, end]


def build_result(
    image_path: Path,
    simplified: list[tuple[float, float]],
    bounds: tuple[int, int, int, int],
    x_calibration: Calibration,
    y_calibration: Calibration,
) -> dict:
    x0, y0, _, _ = bounds
    points = []

    for x, y in simplified:
        full_x = x0 + x
        full_y = y0 + y
        station_ft = x_calibration.value_at(full_x)
        height = y_calibration.value_at(full_y)
        points.append(
            {
                "station": format_station(station_ft),
                "station_ft": round(float(station_ft), 2),
                "height": round(float(height), 2),
            }
        )

    return {
        "source": str(image_path),
        "x_axis": {"unit": "station_ft", "labels_used": x_calibration.labels_used},
        "y_axis": {"unit": "height", "labels_used": y_calibration.labels_used},
        "points": points,
    }


def draw_overlay(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    grid: np.ndarray,
    centerline: list[tuple[float, float]],
    simplified: list[tuple[float, float]],
) -> np.ndarray:
    overlay = image.copy()
    x0, y0, w, h = bounds

    grid_overlay = overlay.copy()
    grid_pixels = np.column_stack(np.where(grid > 0))
    for gy, gx in grid_pixels:
        px = x0 + int(gx)
        py = y0 + int(gy)
        if 0 <= px < overlay.shape[1] and 0 <= py < overlay.shape[0]:
            grid_overlay[py, px] = (0, 255, 0)
    overlay = cv2.addWeighted(grid_overlay, 0.35, overlay, 0.65, 0)

    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 3)

    raw = [(int(round(x0 + x)), int(round(y0 + y))) for x, y in centerline[:: max(1, len(centerline) // 1200)]]
    for p1, p2 in zip(raw, raw[1:]):
        cv2.line(overlay, p1, p2, (0, 180, 255), 2)

    simplified_points = [(int(round(x0 + x)), int(round(y0 + y))) for x, y in simplified]
    for p1, p2 in zip(simplified_points, simplified_points[1:]):
        cv2.line(overlay, p1, p2, (0, 0, 255), 4)
    for point in simplified_points:
        cv2.circle(overlay, point, 8, (0, 0, 255), -1)

    return overlay


def crop_text_from_image(image: np.ndarray, tokens: list[Token], padding: int = 2) -> np.ndarray:
    cleaned = image.copy()
    h, w = image.shape[:2]

    for token in tokens:
        left = max(0, token.left - padding)
        top = max(0, token.top - padding)
        right = min(w - 1, token.left + token.width + padding)
        bottom = min(h - 1, token.top + token.height + padding)
        if right > left and bottom > top:
            cv2.rectangle(cleaned, (left, top), (right, bottom), 255, -1)

    return cleaned


def parse_profile_image(image_path: Path, debug_dir: Path | None, epsilon: float) -> dict:
    image, gray = load_image(image_path)
    image, gray = enhance_profile_contrast(image, gray)
    bounds = find_plot_bounds(gray)
    x0, y0, w, h = bounds
    crop_gray = gray[y0 : y0 + h, x0 : x0 + w]

    tokens = ocr_tokens(image)
    x_calibration = calibrate_x(tokens, bounds)
    y_calibration = calibrate_y(tokens, bounds)
    image_text = get_ocr_tokens(crop_gray, min_confidence=TEXT_REMOVAL_MIN_CONFIDENCE)
    image_without_text = crop_text_from_image(crop_gray, image_text)

    grid, grid_rows, grid_cols = detect_grid(image_without_text)
    line, closed = extract_line_mask(image_without_text, grid, tokens, bounds)
    centerline = trace_centerline(line)
    simplified = rdp(centerline, epsilon=epsilon)

    result = build_result(image_path, simplified, bounds, x_calibration, y_calibration)

    write_debug(debug_dir, "01_plot_crop.png", image[y0 : y0 + h, x0 : x0 + w])
    write_debug(debug_dir, "00_text_removed_crop.png", image_without_text)
    write_debug(debug_dir, "02_grid_mask.png", grid)
    write_debug(debug_dir, "03_line_mask.png", line)
    write_debug(debug_dir, "04_closed_candidates.png", closed)
    write_debug(debug_dir, "06_overlay.png", draw_overlay(image, bounds, grid, centerline, simplified))

    if debug_dir:
        meta = {
            "bounds": {"x": x0, "y": y0, "width": w, "height": h},
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "ocr_tokens": [token.__dict__ for token in tokens],
            "crop_ocr_tokens": [token.__dict__ for token in image_text],
        }
        (debug_dir / "debug.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return result


def merge_profile_results(results: list[dict]) -> dict:
    if not results:
        raise ValueError("At least one profile result is required")
    if len(results) == 1:
        return results[0]

    merged_points = []
    profiles = []
    station_offset = 0.0

    for index, result in enumerate(results):
        points = result["points"]
        if len(points) < 2:
            raise ValueError(f"Profile {index + 1} has fewer than two points")

        source_start = float(points[0]["station_ft"])
        source_end = float(points[-1]["station_ft"])
        adjusted_start = station_offset
        adjusted_end = station_offset + (source_end - source_start)

        profiles.append(
            {
                "source": result["source"],
                "source_station_range_ft": [round(source_start, 2), round(source_end, 2)],
                "merged_station_range_ft": [round(adjusted_start, 2), round(adjusted_end, 2)],
                "point_count": len(points),
            }
        )

        for point_index, point in enumerate(points):
            if index > 0 and point_index == 0:
                continue

            station_ft = station_offset + (float(point["station_ft"]) - source_start)
            merged_points.append(
                {
                    "station": format_station(station_ft),
                    "station_ft": round(station_ft, 2),
                    "height": point["height"],
                    "source_profile_index": index,
                    "source_station": point["station"],
                    "source_station_ft": point["station_ft"],
                }
            )

        station_offset = adjusted_end

    return {
        "source": [result["source"] for result in results],
        "profile_count": len(results),
        "profiles": profiles,
        "x_axis": {
            "unit": "station_ft",
            "labels_used": interpolated_station_labels(
                [format_station(0), format_station(merged_points[-1]["station_ft"])]
            ),
        },
        "y_axis": {
            "unit": "height",
            "labels_used": sorted(
                {
                    label
                    for result in results
                    for label in result.get("y_axis", {}).get("labels_used", [])
                },
                key=lambda label: float(label),
            ),
        },
        "points": merged_points,
    }


def parse_profiles(image_paths: list[Path], output_path: Path, debug_dir: Path | None, epsilon: float) -> dict:
    if not image_paths:
        raise ValueError("At least one profile image is required")

    results = []
    for index, image_path in enumerate(image_paths):
        if len(image_paths) == 1:
            image_debug_dir = debug_dir
        elif debug_dir is None:
            image_debug_dir = None
        else:
            image_debug_dir = debug_dir / f"profile-{index + 1:02d}"
        results.append(parse_profile_image(image_path, image_debug_dir, epsilon))

    result = merge_profile_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_profile(image_path: Path, output_path: Path, debug_dir: Path | None, epsilon: float) -> dict:
    return parse_profiles([image_path], output_path, debug_dir, epsilon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse an engineering profile drawing into station/elevation points.")
    parser.add_argument("images", nargs="+", type=Path, help="Input profile image(s), in route order")
    parser.add_argument("--out", type=Path, default=Path("assets/points.json"), help="Output JSON path")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory for debug masks and overlays")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0,
        help="Ramer-Douglas-Peucker simplification tolerance in pixels",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = parse_profiles(args.images, args.out, args.debug_dir, args.epsilon)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(result['points'])} points to {args.out}")
    if args.debug_dir:
        print(f"Debug artifacts saved to {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
