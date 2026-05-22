import json
import tempfile
import unittest
from pathlib import Path

import profile_to_points


class ProfileParserTest(unittest.TestCase):
    def test_station_format(self):
        self.assertEqual(profile_to_points.format_station(0), "0+00")
        self.assertEqual(profile_to_points.format_station(263), "2+63")
        self.assertEqual(profile_to_points.format_station(1000), "10+00")

    def test_interpolated_station_labels(self):
        labels = profile_to_points.interpolated_station_labels(["0+00", "2+00", "4+00", "5+00"])
        self.assertEqual(labels, ["0+00", "1+00", "2+00", "3+00", "4+00", "5+00"])

    def test_pipe_profile_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "points.json"
            result = profile_to_points.parse_profile(Path("assets/pipe.png"), output, None, epsilon=8.0)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, saved)
        self.assertEqual(
            result["x_axis"]["labels_used"],
            [
                "0+00",
                "1+00",
                "2+00",
                "3+00",
                "4+00",
                "5+00",
                "6+00",
                "7+00",
                "8+00",
                "9+00",
                "10+00",
                "11+00",
                "12+00",
            ],
        )
        self.assertGreaterEqual(len(result["points"]), 6)
        self.assertLessEqual(len(result["points"]), 25)

        stations = [point["station_ft"] for point in result["points"]]
        heights = [point["height"] for point in result["points"]]
        self.assertEqual(stations, sorted(stations))
        self.assertTrue(all(90 <= height <= 130 for height in heights))

        min_point = min(result["points"], key=lambda point: point["height"])
        self.assertGreaterEqual(min_point["station_ft"], 200)
        self.assertLessEqual(min_point["station_ft"], 350)

    def test_merge_profiles_continues_stationing(self):
        first = {
            "source": "first.png",
            "x_axis": {"unit": "station_ft", "labels_used": ["0+00", "1+00"]},
            "y_axis": {"unit": "height", "labels_used": ["100", "110"]},
            "points": [
                {"station": "0+00", "station_ft": 0.0, "height": 100.0},
                {"station": "1+00", "station_ft": 100.0, "height": 101.0},
            ],
        }
        second = {
            "source": "second.png",
            "x_axis": {"unit": "station_ft", "labels_used": ["0+00", "1+00"]},
            "y_axis": {"unit": "height", "labels_used": ["100", "110"]},
            "points": [
                {"station": "0+00", "station_ft": 0.0, "height": 101.0},
                {"station": "0+50", "station_ft": 50.0, "height": 102.0},
                {"station": "1+00", "station_ft": 100.0, "height": 103.0},
            ],
        }

        merged = profile_to_points.merge_profile_results([first, second])

        self.assertEqual([point["station_ft"] for point in merged["points"]], [0.0, 100.0, 150.0, 200.0])
        self.assertEqual([point["station"] for point in merged["points"]], ["0+00", "1+00", "1+50", "2+00"])
        self.assertEqual(merged["profiles"][1]["merged_station_range_ft"], [100.0, 200.0])


if __name__ == "__main__":
    unittest.main()
