# Third-party code

This project as a whole is licensed under the PolyForm Noncommercial License 1.0.0 (see `LICENSE.md`). The
portions listed below were adapted from MIT-licensed projects; their original copyright and permission notices
are reproduced in `third_party/` and continue to apply to those portions.

- **jev-visual** (MIT, © 2026 Jev Visual contributors, https://github.com/hr98w/jev-visual):
  prompt construction, candidate tokenization checks, label / single-token / sequence scoring,
  typed answer assembly and the shared-prefix design were ported from its MLX implementation to
  PyTorch (`jev_photo/jev/{preprocessing,scoring,schema}.py`, `adapter_torch.py`).
  License: [third_party/JEV-VISUAL-LICENSE.txt](third_party/JEV-VISUAL-LICENSE.txt).
- **jev-numeric** (MIT, © 2026 Bring-AI contributors, https://github.com/Bring-AI/jev-numeric):
  the partition → choose → refine interval algorithm behind `number` questions
  (`jev_photo/jev/numeric.py`, `decoding="interval"`).
  License: [third_party/JEV-NUMERIC-LICENSE.txt](third_party/JEV-NUMERIC-LICENSE.txt).
- **Qwen3.5-9B** weights (Qwen team, Apache-2.0) are downloaded from Hugging Face, not redistributed.
- **Example photos** in `examples/photos/` (and their thumbnails in `docs/showcase/`) are CC0 images from
  Wikimedia Commons; see [examples/photos/SOURCES.md](examples/photos/SOURCES.md).
