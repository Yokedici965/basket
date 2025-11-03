from __future__ import annotations

from pathlib import Path

from tools.render_overlays import load_court_shapes


def test_load_court_shapes_parses_polygon_and_hoops(tmp_path: Path):
    yaml_content = """court:
  polygon:
    - [0, 0]
    - [10, 0]
    - [10, 10]
    - [0, 10]
  hoops:
    - name: left
      x: 5.5
      y: 8.2
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_content, encoding="utf-8")

    polygon, hoops = load_court_shapes(cfg_path)
    assert polygon == [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert hoops == [("left", 5.5, 8.2)]

