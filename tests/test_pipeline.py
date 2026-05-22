import tempfile
import unittest
from shutil import copyfile
from pathlib import Path
from unittest.mock import patch

import pipeline


class PipelineTest(unittest.TestCase):
    @patch("pipeline.gemini_image_edit.edit_profile_image")
    @patch("pipeline.gemini_image_edit.edit_plan_image")
    def test_full_pipeline_smoke(self, edit_plan_image, edit_profile_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            def fake_gemini_edit(input_image, output_image, model):
                output_image.parent.mkdir(parents=True, exist_ok=True)
                copyfile(input_image, output_image)
                return output_image

            edit_plan_image.side_effect = fake_gemini_edit
            edit_profile_image.side_effect = fake_gemini_edit
            summary = pipeline.run_pipeline(
                profile_image=[Path("assets/pipe.png")],
                plan_image=Path("assets/img.png"),
                profile_json=tmp / "points.json",
                pipe_3d_json=tmp / "pipe_3d.json",
                obj_output=tmp / "pipe.obj",
                diameter_ft=0.5,
                debug_dir=tmp / "debug",
                profile_epsilon=8.0,
                sample_ft=10.0,
                plan_simplify_px=2.0,
                obj_segments=8,
                cap_ends=True,
                object_name="pipeline_pipe",
            )

            edit_plan_image.assert_called_once()
            edit_profile_image.assert_called_once()
            self.assertTrue((tmp / "points.json").exists())
            self.assertTrue((tmp / "pipe_3d.json").exists())
            self.assertTrue((tmp / "pipe.obj").exists())
            self.assertTrue((tmp / "debug" / "gemini" / "profile.png").exists())
            self.assertTrue((tmp / "debug" / "gemini" / "plan.png").exists())
            self.assertTrue((tmp / "debug" / "summary.json").exists())

        self.assertEqual(summary["profile_image_original"], ["assets/pipe.png"])
        self.assertEqual(summary["profile_image_used"], [str(tmp / "debug" / "gemini" / "profile.png")])
        self.assertEqual(summary["plan_image_used"], str(tmp / "debug" / "gemini" / "plan.png"))
        self.assertEqual(summary["plan_image_original"], "assets/img.png")
        self.assertTrue(summary["gemini"]["enabled"])
        self.assertGreaterEqual(summary["profile_points"], 6)
        self.assertGreaterEqual(summary["pipe_3d_points"], 20)
        self.assertGreater(summary["obj_vertices"], 0)
        self.assertGreater(summary["obj_faces"], 0)


if __name__ == "__main__":
    unittest.main()
