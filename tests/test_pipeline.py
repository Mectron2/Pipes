import tempfile
import unittest
from shutil import copyfile
from pathlib import Path
from unittest.mock import patch

import pipeline


class PipelineTest(unittest.TestCase):
    def test_next_run_dir_uses_next_numeric_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir) / "runs"
            (runs / "run_1").mkdir(parents=True)
            (runs / "run_3").mkdir()
            (runs / "notes").mkdir()

            self.assertEqual(pipeline.next_run_dir(runs), runs / "run_4")

    def test_write_pipe_coordinates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pipe_coordinates.csv"
            pipeline.write_pipe_coordinates_csv(
                {
                    "points": [
                        {
                            "index": 0,
                            "station": "0+00",
                            "chainage_ft": 0.0,
                            "x_px": 1.0,
                            "y_px": 2.0,
                            "x_ft": 3.0,
                            "y_ft": 4.0,
                            "z_ft": 5.0,
                            "height": 5.0,
                        }
                    ]
                },
                output,
            )

            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [
                    "index,station,chainage_ft,x_px,y_px,x_ft,y_ft,z_ft,height",
                    "0,0+00,0.0,1.0,2.0,3.0,4.0,5.0,5.0",
                ],
            )

    @patch("pipeline.pipe_json_to_obj.convert_json_to_obj")
    @patch("pipeline.plan_to_3d.build_pipe_3d")
    @patch("pipeline.profile_parser.parse_profiles")
    @patch("pipeline.gemini_image_edit.edit_profile_image_async")
    @patch("pipeline.gemini_image_edit.edit_plan_image_async")
    def test_pipeline_passes_gemini_outputs_to_analysis(
        self,
        edit_plan_image,
        edit_profile_image,
        parse_profiles,
        build_pipe_3d,
        convert_json_to_obj,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profile_input = tmp / "run_1" / "input" / "profile.png"
            plan_input = tmp / "run_1" / "input" / "plan.png"
            profile_gemini = tmp / "run_1" / "gemini_analized" / "profile.png"
            plan_gemini = tmp / "run_1" / "gemini_analized" / "plan.png"
            profile_json = tmp / "run_1" / "output" / "profile_points.json"
            pipe_3d_json = tmp / "run_1" / "output" / "pipe_3d.json"
            pipe_csv = tmp / "run_1" / "output" / "pipe_coordinates.csv"
            obj_output = tmp / "run_1" / "output" / "pipe.obj"
            debug_dir = tmp / "run_1" / "debug"

            async def fake_profile_edit(input_image, output_image, model):
                self.assertEqual(input_image, profile_input)
                self.assertEqual(output_image, profile_gemini)
                return output_image

            async def fake_plan_edit(input_image, output_image, model):
                self.assertEqual(input_image, plan_input)
                self.assertEqual(output_image, plan_gemini)
                return output_image

            edit_profile_image.side_effect = fake_profile_edit
            edit_plan_image.side_effect = fake_plan_edit
            parse_profiles.return_value = {"points": [{"station_ft": 0.0, "height": 100.0}]}
            build_pipe_3d.return_value = {
                "points": [
                    {
                        "index": 0,
                        "station": "0+00",
                        "chainage_ft": 0.0,
                        "x_px": 10.0,
                        "y_px": 20.0,
                        "x_ft": 0.0,
                        "y_ft": 0.0,
                        "z_ft": 100.0,
                        "height": 100.0,
                    }
                ]
            }
            convert_json_to_obj.return_value = (8, 6)

            summary = pipeline.run_pipeline(
                profile_image=[profile_input],
                plan_image=plan_input,
                profile_json=profile_json,
                pipe_3d_json=pipe_3d_json,
                obj_output=obj_output,
                diameter_ft=0.5,
                debug_dir=debug_dir,
                profile_epsilon=8.0,
                sample_ft=10.0,
                plan_simplify_px=2.0,
                obj_segments=8,
                cap_ends=True,
                object_name="pipeline_pipe",
                pipe_csv_output=pipe_csv,
                gemini_profile_images=[profile_gemini],
                gemini_plan_image=plan_gemini,
            )

            parse_profiles.assert_called_once_with([profile_gemini], profile_json, debug_dir / "profile", 8.0)
            build_pipe_3d.assert_called_once_with(plan_gemini, profile_json, pipe_3d_json, debug_dir / "plan-3d", 10.0, 2.0)
            self.assertTrue(pipe_csv.exists())
            self.assertEqual(summary["outputs"]["pipe_coordinates_csv"], str(pipe_csv))

    @unittest.skipUnless(
        Path("assets/pipe.png").exists() and Path("assets/img.png").exists(),
        "requires assets/pipe.png and assets/img.png fixtures",
    )
    @patch("pipeline.gemini_image_edit.edit_profile_image_async")
    @patch("pipeline.gemini_image_edit.edit_plan_image_async")
    def test_full_pipeline_smoke(self, edit_plan_image, edit_profile_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            async def fake_gemini_edit(input_image, output_image, model):
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
