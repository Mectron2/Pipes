import csv
import tempfile
import unittest
from pathlib import Path

import pipe_top_side_csv


class PipeTopSideCsvTest(unittest.TestCase):
    def test_build_top_side_rows_uses_foot_columns(self):
        pipe_3d = {
            "points": [
                {"chainage_ft": 0.0, "x_ft": 0.0, "y_ft": 0.0, "z_ft": 100.0},
                {"chainage_ft": 10.0, "x_ft": 10.0, "y_ft": 0.0, "z_ft": 99.5},
                {"chainage_ft": 20.0, "x_ft": 10.0, "y_ft": 10.0, "z_ft": 99.0},
            ]
        }

        rows = pipe_top_side_csv.build_top_side_rows(pipe_3d, pipe_od_mm=426, bend_angle_degrees=10)

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["view"], "TOP")
        self.assertEqual(rows[0]["point_name"], "T01")
        self.assertEqual(rows[0]["cad_x_ft"], "0.000")
        self.assertEqual(rows[0]["station_ft"], "0.000")
        self.assertEqual(rows[0]["elevation_ft"], "100.000")
        self.assertEqual(rows[0]["pipe_od_mm"], "426")
        self.assertEqual(rows[1]["segment_type"], "bend")
        self.assertEqual(rows[3]["view"], "SIDE")
        self.assertEqual(rows[3]["point_name"], "S01")
        self.assertEqual(rows[3]["cad_x_ft"], "0.000")
        self.assertEqual(rows[3]["cad_y_ft"], "100.000")
        self.assertEqual(rows[3]["segment_type"], "profile")

    def test_write_top_side_csv(self):
        pipe_3d = {
            "points": [
                {"chainage_ft": 0.0, "x_ft": 0.0, "y_ft": 0.0, "z_ft": 100.0},
                {"chainage_ft": 10.0, "x_ft": 10.0, "y_ft": 0.0, "z_ft": 99.5},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pipe.csv"
            pipe_top_side_csv.write_top_side_csv(pipe_3d, output, pipe_od_mm=426)
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(pipe_top_side_csv.CSV_FIELDS, list(rows[0].keys()))
        self.assertEqual([row["view"] for row in rows], ["TOP", "TOP", "SIDE", "SIDE"])
        self.assertEqual(rows[-1]["point_name"], "S02")

    def test_run_top_side_csv_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = pipe_top_side_csv.run_top_side_csv(
                profile_images=[Path("assets/profile-2.png")],
                plan_image=Path("assets/top-2-highlighted.png"),
                output_csv=tmp / "pipe_baseline_top_side.csv",
                pipe_od_mm=426,
                profile_json=tmp / "points.json",
                pipe_3d_json=tmp / "pipe_3d.json",
                debug_dir=None,
                profile_epsilon=8.0,
                sample_ft=100.0,
                plan_simplify_px=2.0,
                bend_angle_degrees=10.0,
            )
            with (tmp / "pipe_baseline_top_side.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(summary["csv_rows"], len(rows))
        self.assertGreater(summary["profile_points"], 1)
        self.assertGreater(summary["pipe_3d_points"], 1)
        self.assertEqual(rows[0]["view"], "TOP")
        self.assertEqual(rows[summary["pipe_3d_points"]]["view"], "SIDE")
        self.assertEqual(rows[0]["point_name"], "T01")
        self.assertEqual(rows[summary["pipe_3d_points"]]["point_name"], "S01")
        self.assertEqual(
            [float(row["station_ft"]) for row in rows if row["view"] == "TOP"],
            sorted(float(row["station_ft"]) for row in rows if row["view"] == "TOP"),
        )


if __name__ == "__main__":
    unittest.main()
