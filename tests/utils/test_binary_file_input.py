from pathlib import Path
import numpy as np
import shutil
import pickle

import unittest

from dpgen2.utils import BinaryFileInput


class TestBinaryFileInput(unittest.TestCase):
    def setUp(self):
        self.task_input_path = Path("task/input")
        self.task_input_path.mkdir(parents=True, exist_ok=True)

        self.task_output_path = Path("task/output")
        self.task_output_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if Path("task").is_dir():
            shutil.rmtree("task")

    def test_read_file(self):
        (self.task_input_path / "input_file.txt").write_text("foo")
        (self.task_input_path / "input_file").write_text("foo")

        txt = BinaryFileInput(self.task_input_path / "input_file.txt")
        self.assertIsInstance(txt._data, bytes)
        self.assertTrue(len(txt._data) > 0)

        txt = BinaryFileInput(self.task_input_path / "input_file")
        self.assertIsInstance(txt._data, bytes)
        self.assertTrue(len(txt._data) > 0)

        txt = BinaryFileInput(self.task_input_path / "input_file.txt", "txt")
        self.assertIsInstance(txt._data, bytes)
        self.assertTrue(len(txt._data) > 0)

        txt = BinaryFileInput(self.task_input_path / "input_file.txt", ".txt")

        self.assertRaisesRegex(
            AssertionError,
            "File extension mismatch",
            BinaryFileInput,
            self.task_input_path / "input_file.txt",
            "jpg",
        )

        self.assertRaisesRegex(
            AssertionError,
            "File extension mismatch",
            BinaryFileInput,
            self.task_input_path / "input_file.txt",
            "tx",
        )

        self.assertRaisesRegex(
            AssertionError,
            "File extension mismatch",
            BinaryFileInput,
            self.task_input_path / "input_file.txt",
            "t",
        )

        self.assertRaisesRegex(
            AssertionError,
            "File extension mismatch",
            BinaryFileInput,
            self.task_input_path / "input_file",
            "e",
        )

        self.assertRaisesRegex(
            AssertionError,
            "File extension mismatch",
            BinaryFileInput,
            self.task_input_path / "input_file",
            "file",
        )

    def test_save_file(self):
        (self.task_input_path / "input_file.txt").write_text("foo")
        (self.task_input_path / "input_file").write_text("foo")

        txt_with_ext = BinaryFileInput(self.task_input_path / "input_file.txt", "txt")
        txt = BinaryFileInput(self.task_input_path / "input_file.txt")

        self.assertWarnsRegex(
            UserWarning,
            "file extension mismatch!",
            txt_with_ext.save_as_file,
            self.task_output_path / "output_file",
        )
        self.assertEqual(Path(self.task_output_path / "output_file").read_text(), "foo")

        self.assertWarnsRegex(
            UserWarning,
            "file extension mismatch!",
            txt_with_ext.save_as_file,
            self.task_output_path / "output_file.t",
        )
        self.assertEqual(
            Path(self.task_output_path / "output_file.t").read_text(), "foo"
        )

        txt_with_ext.save_as_file(self.task_output_path / "output_file.txt")
        self.assertEqual(
            Path(self.task_output_path / "output_file.txt").read_text(), "foo"
        )

        txt.save_as_file(self.task_output_path / "output_file1.txt")
        self.assertEqual(
            Path(self.task_output_path / "output_file1.txt").read_text(), "foo"
        )

        txt.save_as_file(self.task_output_path / "output_file1.jpg")
        self.assertEqual(
            Path(self.task_output_path / "output_file1.jpg").read_text(), "foo"
        )

    def test_serialization(self):
        def serialization(obj):
            with open(self.task_output_path / "tmp.pkl", "wb") as f:
                pickle.dump(obj, f)

            with open(self.task_output_path / "tmp.pkl", "rb") as f:
                return pickle.load(f)

        # check text file
        (self.task_input_path / "input_file.txt").write_text("foo")
        txt = BinaryFileInput(self.task_input_path / "input_file.txt", "txt")
        txt = serialization(txt)

        txt.save_as_file(self.task_output_path / "output_file.txt")
        self.assertEqual(
            Path(self.task_output_path / "output_file.txt").read_text(), "foo"
        )

        # check binary file
        tensor = np.random.random((3, 2))
        np.save(self.task_output_path / "tensor.npy", tensor)
        t = BinaryFileInput(self.task_output_path / "tensor.npy", "npy")
        t = serialization(t)

        self.assertWarnsRegex(
            UserWarning,
            "file extension mismatch!",
            t.save_as_file,
            self.task_output_path / "output_file.py",
        )
        with open(Path(self.task_output_path / "output_file.py"), "rb") as f:
            _tensor = np.load(f)
        self.assertTrue(np.allclose(tensor, _tensor))

        t.save_as_file(self.task_output_path / "output_file1.npy")
        with open(Path(self.task_output_path / "output_file1.npy"), "rb") as f:
            _tensor = np.load(f)
        self.assertTrue(np.allclose(tensor, _tensor))
