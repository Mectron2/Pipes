import json
import tempfile
import unittest
from pathlib import Path

import plan_to_3d


class PlanTo3DTest(unittest.TestCase):
    def test_interpolate_height(self):
        profile = [
            {"station_ft": 0.0, "height": 100.0},
            {"station_ft": 100.0, "height": 110.0},
        ]
        self.assertEqual(plan_to_3d.interpolate_height(profile, -10), 100.0)
        self.assertEqual(plan_to_3d.interpolate_height(profile, 0), 100.0)
        self.assertEqual(plan_to_3d.interpolate_height(profile, 50), 105.0)
        self.assertEqual(plan_to_3d.interpolate_height(profile, 100), 110.0)
        self.assertEqual(plan_to_3d.interpolate_height(profile, 150), 110.0)

    def test_build_3d_result_maps_endpoints(self):
        centerline = [(0.0, 0.0), (10.0, 0.0)]
        profile = [
            {"station_ft": 0.0, "height": 100.0},
            {"station_ft": 100.0, "height": 110.0},
        ]
        result = plan_to_3d.build_3d_result(Path("plan.png"), Path("profile.json"), centerline, profile, sample_ft=50)

        self.assertEqual([point["chainage_ft"] for point in result["points"]], [0.0, 50.0, 100.0])
        self.assertEqual([point["x_px"] for point in result["points"]], [0.0, 5.0, 10.0])
        self.assertEqual([point["height"] for point in result["points"]], [100.0, 105.0, 110.0])
        self.assertEqual(result["plan_ft_per_px"], 10.0)
        self.assertEqual([point["x_ft"] for point in result["points"]], [0.0, 50.0, 100.0])
        self.assertEqual([point["y_ft"] for point in result["points"]], [0.0, 0.0, 0.0])
        self.assertEqual([point["z_ft"] for point in result["points"]], [100.0, 105.0, 110.0])

    @unittest.skipUnless(
        Path("assets/img.png").exists() and Path("assets/points.json").exists(),
        "requires assets/img.png and assets/points.json fixtures",
    )
    def test_img_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pipe_3d.json"
            result = plan_to_3d.build_pipe_3d(
                Path("assets/img.png"),
                Path("assets/points.json"),
                output,
                None,
                sample_ft=10.0,
                simplify_px=2.0,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, saved)
        self.assertGreaterEqual(len(result["points"]), 20)
        self.assertEqual(result["xy_unit"], "pixel")
        self.assertEqual(result["xyz_unit"], "ft")
        self.assertGreater(result["plan_ft_per_px"], 0)
        self.assertGreater(result["plan_px_per_ft"], 0)

        chainages = [point["chainage_ft"] for point in result["points"]]
        self.assertEqual(chainages, sorted(chainages))
        self.assertAlmostEqual(chainages[0], result["station_range_ft"][0])
        self.assertAlmostEqual(chainages[-1], result["station_range_ft"][1])

        heights = [point["height"] for point in result["points"]]
        self.assertTrue(all(90 <= height <= 130 for height in heights))
        self.assertEqual(heights, [point["z_ft"] for point in result["points"]])

        first = result["points"][0]
        last = result["points"][-1]
        self.assertLess(first["x_px"], last["x_px"])
        self.assertGreater(first["y_px"], last["y_px"])
        self.assertEqual(first["x_ft"], 0.0)
        self.assertEqual(first["y_ft"], 0.0)
        self.assertGreater(last["x_ft"], first["x_ft"])


if __name__ == "__main__":
    unittest.main()
