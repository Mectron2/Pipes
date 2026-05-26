import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import pipeline


class PipelineTest(unittest.TestCase):
    def test_parse_args_accepts_output_dir(self):
        with mock.patch(
            "sys.argv",
            ["pipeline.py", "--diameter-ft", "0.5", "--output-dir", "custom-output"],
        ):
            args = pipeline.parse_args()

        self.assertEqual(args.output_dir, Path("custom-output"))

    def test_next_run_dir_uses_next_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            (output_root / "run_1").mkdir(parents=True)
            (output_root / "run_7").mkdir()
            (output_root / "run_draft").mkdir()

            self.assertEqual(pipeline.next_run_dir(output_root), output_root / "run_8")

    def test_resolve_cli_paths_defaults_to_next_run_under_default_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            (output_root / "run_1").mkdir(parents=True)
            args = SimpleNamespace(
                output_dir=None,
                profile_json=None,
                pipe_3d_json=None,
                obj=None,
                csv_output=None,
                debug_dir=None,
            )

            with mock.patch.object(pipeline, "DEFAULT_OUTPUT_DIR", output_root):
                resolved = pipeline.resolve_cli_paths(args)

            self.assertEqual(resolved.output_dir, output_root / "run_2")
            self.assertEqual(resolved.profile_json, output_root / "run_2" / "points.json")
            self.assertEqual(resolved.pipe_3d_json, output_root / "run_2" / "pipe_3d.json")
            self.assertEqual(resolved.obj, output_root / "run_2" / "pipe.obj")
            self.assertEqual(resolved.csv_output, output_root / "run_2" / "pipe_baseline_top_side.csv")
            self.assertEqual(resolved.debug_dir, output_root / "run_2" / "debug-pipeline")

    def test_resolve_cli_paths_keeps_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = SimpleNamespace(
                output_dir=tmp / "custom-output",
                profile_json=tmp / "profile.json",
                pipe_3d_json=tmp / "pipe.json",
                obj=tmp / "pipe.obj",
                csv_output=tmp / "pipe.csv",
                debug_dir=tmp / "debug",
            )

            resolved = pipeline.resolve_cli_paths(args)

            self.assertEqual(resolved.output_dir, tmp / "custom-output")
            self.assertEqual(resolved.profile_json, tmp / "profile.json")
            self.assertEqual(resolved.pipe_3d_json, tmp / "pipe.json")
            self.assertEqual(resolved.obj, tmp / "pipe.obj")
            self.assertEqual(resolved.csv_output, tmp / "pipe.csv")
            self.assertEqual(resolved.debug_dir, tmp / "debug")

    def test_resolve_cli_paths_uses_output_dir_for_default_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "custom-output"
            args = SimpleNamespace(
                output_dir=output_dir,
                profile_json=None,
                pipe_3d_json=None,
                obj=None,
                csv_output=None,
                debug_dir=None,
            )

            resolved = pipeline.resolve_cli_paths(args)

            self.assertEqual(resolved.output_dir, output_dir)
            self.assertEqual(resolved.profile_json, output_dir / "points.json")
            self.assertEqual(resolved.pipe_3d_json, output_dir / "pipe_3d.json")
            self.assertEqual(resolved.obj, output_dir / "pipe.obj")
            self.assertEqual(resolved.csv_output, output_dir / "pipe_baseline_top_side.csv")
            self.assertEqual(resolved.debug_dir, output_dir / "debug-pipeline")

    def test_run_pipeline_writes_outputs_to_requested_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profile_json = tmp / "points.json"
            pipe_3d_json = tmp / "pipe_3d.json"
            obj_output = tmp / "pipe.obj"
            csv_output = tmp / "pipe_baseline_top_side.csv"

            def parse_profiles_side_effect(_profile_images, output_path, _debug_dir, _epsilon):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text('{"points": [{"station_ft": 0}]}', encoding="utf-8")
                return {"points": [{"station_ft": 0}]}

            def build_pipe_3d_side_effect(_plan_image, _profile_json, output_path, *_args, **_kwargs):
                pipe_3d = {"points": [{"chainage_ft": 0.0, "x_ft": 0.0, "y_ft": 0.0, "z_ft": 100.0}]}
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text('{"points": []}', encoding="utf-8")
                return pipe_3d

            def convert_pipe_data_to_obj_side_effect(_pipe_3d, output_path, *_args):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("o pipeline_pipe\n", encoding="utf-8")
                return {"vertices": [], "faces": [], "vertex_count": 8, "face_count": 6}

            def write_top_side_csv_side_effect(_pipe_3d, output_path, *_args):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("view\nTOP\n", encoding="utf-8")
                return [{"view": "TOP"}]

            with (
                mock.patch.object(pipeline.profile_parser, "parse_profiles", side_effect=parse_profiles_side_effect),
                mock.patch.object(pipeline.plan_to_3d, "build_pipe_3d", side_effect=build_pipe_3d_side_effect),
                mock.patch.object(
                    pipeline.pipe_json_to_obj,
                    "convert_pipe_data_to_obj",
                    side_effect=convert_pipe_data_to_obj_side_effect,
                ),
                mock.patch.object(
                    pipeline.pipe_top_side_csv,
                    "write_top_side_csv",
                    side_effect=write_top_side_csv_side_effect,
                ),
            ):
                summary = pipeline.run_pipeline(
                    profile_image=[Path("input/profile.png")],
                    plan_image=Path("input/top.png"),
                    profile_json=profile_json,
                    pipe_3d_json=pipe_3d_json,
                    obj_output=obj_output,
                    diameter_ft=0.5,
                    debug_dir=tmp / "debug",
                    profile_epsilon=8.0,
                    sample_ft=10.0,
                    plan_simplify_px=2.0,
                    obj_segments=8,
                    cap_ends=True,
                    object_name="pipeline_pipe",
                    csv_output=csv_output,
                )

            self.assertTrue(profile_json.exists())
            self.assertTrue(pipe_3d_json.exists())
            self.assertTrue(obj_output.exists())
            self.assertTrue(csv_output.exists())
            self.assertTrue((tmp / "debug" / "summary.json").exists())

        self.assertEqual(summary["outputs"]["profile_json"], str(profile_json))
        self.assertEqual(summary["outputs"]["pipe_3d_json"], str(pipe_3d_json))
        self.assertEqual(summary["outputs"]["obj"], str(obj_output))
        self.assertEqual(summary["outputs"]["csv"], str(csv_output))
        self.assertEqual(summary["profile_points"], 1)
        self.assertEqual(summary["pipe_3d_points"], 1)
        self.assertGreater(summary["obj_vertices"], 0)
        self.assertGreater(summary["obj_faces"], 0)

    def test_use_gemini_plan_preprocesses_plan_before_3d_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            expected_gemini_plan = tmp / "debug" / "gemini" / "plan.png"
            highlighted_image = np.zeros((2, 2, 3), dtype=np.uint8)
            gemini_highlighter = mock.Mock()
            gemini_highlighter.highlight_force_main_image.return_value = highlighted_image

            with (
                mock.patch.object(pipeline.profile_parser, "parse_profiles", return_value={"points": [{"station_ft": 0}]}),
                mock.patch.dict("sys.modules", {"gemini_highlighter": gemini_highlighter}),
                mock.patch.object(pipeline.plan_to_3d, "build_pipe_3d", return_value={"points": [{"x": 1}]}) as build_pipe_3d,
                mock.patch.object(
                    pipeline.pipe_json_to_obj,
                    "convert_pipe_data_to_obj",
                    return_value={"vertices": [], "faces": [], "vertex_count": 8, "face_count": 6},
                ),
            ):
                summary = pipeline.run_pipeline(
                    profile_image=[Path("assets/profile.png")],
                    plan_image=Path("assets/top.png"),
                    profile_json=tmp / "profile_points.json",
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
                    use_gemini_plan=True,
                )

            gemini_highlighter.highlight_force_main_image.assert_called_once_with(Path("assets/top.png"))
            self.assertTrue(expected_gemini_plan.exists())
            build_pipe_3d.assert_called_once()
            self.assertEqual(build_pipe_3d.call_args.args[0], expected_gemini_plan)
            self.assertEqual(build_pipe_3d.call_args.kwargs["profile_result"], {"points": [{"station_ft": 0}]})
            self.assertIs(build_pipe_3d.call_args.kwargs["plan_image"], highlighted_image)
            self.assertEqual(summary["plan_image"], "assets/top.png")
            self.assertEqual(summary["plan_image_used"], str(expected_gemini_plan))
            self.assertTrue(summary["gemini"]["plan_enabled"])

    def test_csv_export_uses_pipe_3d_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pipe_3d = {
                "points": [
                    {"chainage_ft": 0.0, "x_ft": 0.0, "y_ft": 0.0, "z_ft": 100.0},
                    {"chainage_ft": 10.0, "x_ft": 10.0, "y_ft": 0.0, "z_ft": 99.5},
                ]
            }

            with (
                mock.patch.object(pipeline.profile_parser, "parse_profiles", return_value={"points": [{"station_ft": 0}]}),
                mock.patch.object(pipeline.plan_to_3d, "build_pipe_3d", return_value=pipe_3d),
                mock.patch.object(
                    pipeline.pipe_json_to_obj,
                    "convert_pipe_data_to_obj",
                    return_value={"vertices": [], "faces": [], "vertex_count": 8, "face_count": 6},
                ) as convert_pipe_data_to_obj,
                mock.patch.object(
                    pipeline.pipe_top_side_csv,
                    "write_top_side_csv",
                    return_value=[{"view": "TOP"}, {"view": "SIDE"}],
                ) as write_top_side_csv,
            ):
                summary = pipeline.run_pipeline(
                    profile_image=[Path("assets/profile.png")],
                    plan_image=Path("assets/top.png"),
                    profile_json=tmp / "profile_points.json",
                    pipe_3d_json=tmp / "pipe_3d.json",
                    obj_output=tmp / "pipe.obj",
                    diameter_ft=0.5,
                    debug_dir=None,
                    profile_epsilon=8.0,
                    sample_ft=10.0,
                    plan_simplify_px=2.0,
                    obj_segments=8,
                    cap_ends=True,
                    object_name="pipeline_pipe",
                    csv_output=tmp / "pipe_baseline_top_side.csv",
                )

            expected_pipe_od_mm = 0.5 * pipeline.FEET_TO_MILLIMETERS
            write_top_side_csv.assert_called_once_with(pipe_3d, tmp / "pipe_baseline_top_side.csv", expected_pipe_od_mm, 10.0)
            self.assertIs(convert_pipe_data_to_obj.call_args.args[0], pipe_3d)
            self.assertEqual(summary["outputs"]["csv"], str(tmp / "pipe_baseline_top_side.csv"))
            self.assertEqual(summary["pipe_od_mm"], expected_pipe_od_mm)
            self.assertEqual(summary["csv_rows"], 2)
            self.assertIs(summary["pipe_3d"], pipe_3d)


if __name__ == "__main__":
    unittest.main()
