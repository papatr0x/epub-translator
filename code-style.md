# Code style

## Language
- Code identifiers in English. Comments, docstrings and console output currently in Spanish (keep consistent within a file).

## Structure
- Scripts at the project root: `translate_epub.py` (extract + translate into cache, shared helpers) and `build_epub.py` (builds EPUB from cache; imports helpers from `translate_epub.py`). `fix_epub.py` is obsolete.
- `mlx_lm` is imported lazily inside `Translator`, so `build_epub.py` never loads the model.
- Sections separated by banner comments: `# ─── Name ───...`.
- Design rationale and optimizations are documented in the module docstring of `translate_epub.py` (numbered points); inline comments reference them as "docstring, punto N". Keep both in sync when changing behavior.
- Config as module-level UPPER_CASE constants at the top.
- Paths via `pathlib.Path`, relative to `SCRIPT_DIR`.
- Small top-level functions with type hints on signatures; `main()` parses CLI args with `argparse`.

## Output
- Progress via `print()` with a `[stage]` prefix (`[extract]`, `[translate]`, `[build]`, `[epub]`); `flush=True` for long-running steps.

## Environment / tools
- Python venv in `.venv/` (deps: beautifulsoup4, lxml, mlx_lm). Run with `.venv/bin/python`.
- No tests, linter or build step. Verify by running the script and opening the EPUB in Apple Books.

## Run
```bash
.venv/bin/python -u translate_epub.py input.epub > translation.log 2>&1 &
.venv/bin/python build_epub.py input.epub   # partial or full EPUB from the cache
```
Working files live in `work/<epub name>/` (`src/` extracted original, `cache_*.jsonl` translation progress, `build/` last built EPUB contents).
