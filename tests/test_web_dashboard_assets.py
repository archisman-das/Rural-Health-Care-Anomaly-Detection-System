from __future__ import annotations

from pathlib import Path
import unittest


class WebDashboardAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.web_dir = Path(__file__).resolve().parents[1] / "web"

    def test_static_dashboard_files_exist(self) -> None:
        expected = {
            "index.html": self.web_dir / "index.html",
            "styles.css": self.web_dir / "styles.css",
            "script.js": self.web_dir / "script.js",
        }
        for name, path in expected.items():
            with self.subTest(name=name):
                self.assertTrue(path.exists(), f"Expected {path} to exist")

    def test_index_references_styles_and_script(self) -> None:
        index_html = (self.web_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', index_html)
        self.assertIn('src="script.js"', index_html)
        self.assertIn('id="patient-risk-map"', index_html)
        self.assertIn('id="model-table-body"', index_html)
        self.assertIn('id="runtime-grid"', index_html)

    def test_script_uses_optional_dashboard_data_file(self) -> None:
        script = (self.web_dir / "script.js").read_text(encoding="utf-8")
        self.assertIn('dashboard-data.json', script)
        self.assertIn('loadDashboardData', script)


if __name__ == "__main__":
    unittest.main()
