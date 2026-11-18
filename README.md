# epub-translator

Traduce libros EPUB localmente en Apple Silicon con
[TranslateGemma 4B](https://huggingface.co/mlx-community/translategemma-4b-it-4bit) (MLX).
Gratis y sin enviar el libro a ningún servicio.

- Traduce párrafo por párrafo, conservando la estructura, la cursiva/negrita, los links y las notas.
- Batching continuo: ~0.36 s por párrafo en un M5 Pro (una novela de ~3000 párrafos en ~20 min).
- Retomable: cada traducción se guarda al instante. Si se corta, al relanzar sigue donde quedó.
- EPUB parcial: se puede armar un EPUB con los capítulos ya traducidos mientras la traducción sigue corriendo.

## Requisitos

- Mac con Apple Silicon
- Python 3.10+ (probado con 3.14)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

El modelo (~2 GB) se descarga de Hugging Face la primera vez.

## Uso

Traducir (por defecto alemán → español):

```bash
.venv/bin/python -u translate_epub.py libro.epub > translation.log 2>&1 &
```

Armar el EPUB con lo traducido hasta ahora (completo o parcial):

```bash
.venv/bin/python build_epub.py libro.epub
```

Genera `libro (ES).epub`, o `libro (ES parcial).epub` si la traducción no terminó.

Opciones:

| Opción | Script | Descripción |
|---|---|---|
| `--src de` / `--tgt es` | ambos | Idiomas origen y destino |
| `--batch 16` | `translate_epub.py` | Párrafos en paralelo (32 rinde ~10% más y usa el doble de memoria) |
| `salida.epub` | `build_epub.py` | Ruta del EPUB de salida |

Los archivos de trabajo quedan en `work/<nombre del libro>/`:
`src/` (EPUB original extraído), `cache_<src>-<tgt>.jsonl` (traducciones) y `build/`.

## Cómo funciona

El diseño y las optimizaciones (por qué es rápido y cómo retoma) están explicados
en el docstring de [`translate_epub.py`](translate_epub.py). Las convenciones de
código están en [`code-style.md`](code-style.md).

## Limitaciones

- Calidad de traducción automática: buena para lectura personal, no editorial.
- Si el modelo no respeta la cursiva de un párrafo, ese párrafo pierde el formato (no el texto).
- Fragmentos muy cortos que el modelo no logra traducir se dejan en el idioma original (queda indicado en el log).
