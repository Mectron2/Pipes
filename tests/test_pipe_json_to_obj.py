import tempfile
import unittest
from pathlib import Path

import pipe_json_to_obj


class PipeJsonToObjTest(unittest.TestCase):
    def test_build_tube_mesh_counts(self):
        points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        vertices, faces = pipe_json_to_obj.build_tube_mesh(points, diameter=2.0, radial_segments=8, cap_ends=True)

        self.assertEqual(len(vertices), 26)
        self.assertEqual(len(faces), 32)

    def test_convert_current_pipe_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pipe.obj"
            vertex_count, face_count = pipe_json_to_obj.convert_json_to_obj(
                Path("assets/pipe_3d.json"),
                output,
                diameter_ft=0.5,
                radial_segments=12,
                cap_ends=True,
                object_name="test_pipe",
            )
            content = output.read_text(encoding="utf-8")

        self.assertGreater(vertex_count, 0)
        self.assertGreater(face_count, 0)
        self.assertIn("o test_pipe", content)
        self.assertIn("\nv ", content)
        self.assertIn("\nf ", content)


if __name__ == "__main__":
    unittest.main()
