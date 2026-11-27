# Code style

## Language
- Code identifiers in English. Comments, docstrings and console output currently in Spanish (keep consistent within a file).

## Structure
- Scripts at the project root: `translate_epub.py` (extract + translate into cache, shared helpers) and `build_epub.py` (builds EPUB from cache; imports helpers from `translate_epub.py`). `fix_epub.py` is obsolete.
- `mlx_lm` is imported lazily inside `Translator`, so `build_epub.py` never loads the model.
- `epub_translator_colab.ipynb` is a standalone copy of the shared logic (HTML, cache, notes, build) with a transformers backend. When changing that logic in the scripts, apply the same change to the notebook.
- Models: registry `MODELS` in `translate_epub.py` (`--modelo` alias or MLX repo). Two prompt modes: `translategemma` (its own template) and `chat` (general model with `CHAT_SYSTEM_PROMPT`, thinking disabled). Default model: `gemma4`, batch 32. The model is part of the cache key; the last one used per book is stored in `work/<book>/modelo.txt` and is the default for all scripts.
- Translation notes: per-user `notas/<src>-<tgt>.md` (git-ignored, each user writes their own) plus optional per-book `<book>.notas.md` next to the EPUB. Notes that apply to a text are part of its cache key.
- Sections separated by banner comments: `# ─── Name ───...`.
- Design rationale and optimizations are documented in the module docstring of `translate_epub.py` (numbered points); inline comments reference them as "docstring, punto N". Keep both in sync when changing behavior.
- Config as module-level UPPER_CASE constants at the top.
- Paths via `pathlib.Path`, relative to `SCRIPT_DIR`.
- Small top-level functions with type hints on signatures; `main()` parses CLI args with `argparse`.

## Output
- Progress via `print()` with a `[stage]` prefix (`[extract]`, `[notes]`, `[translate]`, `[build]`, `[epub]`); `flush=True` for long-running steps.

## Environment / tools
- Python venv in `.venv/` (deps: beautifulsoup4, lxml, mlx_lm). Run with `.venv/bin/python`.
- No tests, linter or build step. Verify by running the script and opening the EPUB in Apple Books.

## Run
```bash
.venv/bin/python translate_epub.py input.epub --src de                 # shows plan, asks to confirm
.venv/bin/python -u translate_epub.py input.epub --src de -q > translation.log 2>&1 &   # no prompt (required in background)
.venv/bin/python build_epub.py input.epub --src de   # partial or full EPUB from the cache
```
Working files live in `work/<epub name>/` (`src/` extracted original, `cache_*.jsonl` translation progress, `build/` last built EPUB contents).
