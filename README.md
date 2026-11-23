# epub-translator

Traduce libros EPUB en tu Mac con Apple Silicon usando
[TranslateGemma 4B](https://huggingface.co/mlx-community/translategemma-4b-it-4bit) sobre
[MLX](https://github.com/ml-explore/mlx). Es gratis y todo corre localmente: el libro no se
envía a ningún servicio.

- **Conserva el libro:** traduce párrafo por párrafo y mantiene la estructura, la cursiva y
  la negrita, los links, las notas al pie y el índice.
- **Rápido:** traduce varios párrafos en paralelo. En un M5 Pro, una novela de ~3000 párrafos
  tarda unos 20 minutos.
- **Retomable:** cada párrafo traducido se guarda al instante. Si se corta (Ctrl+C, un cierre
  inesperado), al relanzar sigue donde quedó.
- **EPUB parcial:** puedes armar un EPUB con los capítulos ya traducidos mientras la
  traducción sigue corriendo.

¿No tienes un Mac con Apple Silicon? Usa el [notebook para Google Colab](#google-colab).

## Requisitos

- Mac con Apple Silicon (M1 o posterior). MLX no funciona en Intel, Linux ni Windows.
- Python 3.10 o posterior (probado con 3.14), o [uv](https://docs.astral.sh/uv/).
- Unos 3 GB de memoria libre (medido con el lote por defecto, 16).
- El libro en formato **EPUB**. Un PDF hay que convertirlo antes, por ejemplo con
  [Calibre](https://calibre-ebook.com): `ebook-convert libro.pdf libro.epub`.

## Instalación

Descarga el código:

```bash
git clone https://github.com/papatr0x/epub-translator.git
cd epub-translator
```

Luego crea un entorno virtual con las dependencias. Elige una de las dos opciones.

### Opción A: venv (incluido en Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`source .venv/bin/activate` activa el entorno solo en esa terminal. En una terminal nueva hay
que volver a activarlo (desde la carpeta del proyecto) antes de usar los scripts. Para salir
del entorno: `deactivate`.

### Opción B: uv

```bash
uv venv
uv pip install -r requirements.txt
```

Con uv no hace falta activar nada: `uv run` usa automáticamente el `.venv` de la carpeta.

### Primera ejecución

La primera traducción descarga el modelo de Hugging Face (~2 GB) a `~/.cache/huggingface/`.
No hace falta cuenta. Las siguientes ejecuciones lo usan desde ahí.

## Uso

Los ejemplos usan `python` con el entorno **activado** (opción A). Con **uv** (opción B),
reemplaza `python` por `uv run`:

| venv activado | uv |
|---|---|
| `python translate_epub.py libro.epub` | `uv run translate_epub.py libro.epub` |
| `python build_epub.py libro.epub` | `uv run build_epub.py libro.epub` |

Sin activar el entorno también funciona con la ruta completa: `.venv/bin/python translate_epub.py …`.

El proceso tiene dos pasos independientes: **traducir** (lento, llena una caché) y **armar**
el EPUB (instantáneo, lee la caché). Puedes armar el EPUB tantas veces como quieras.

### 1. Traducir

```bash
python translate_epub.py libro.epub
```

Antes de cargar el modelo muestra qué va a hacer y pregunta si empezar:

```
Plan de traducción
  Libro:      libro.epub (94 archivos HTML)
  Idiomas:    de → es
  Modelo:     mlx-community/translategemma-4b-it-4bit (batch 16)
  Unidades:   3035 en total, 0 ya traducidas, 3035 pendientes
  A traducir: 3050 textos, ~18 min estimados (referencia: 0.36 s/texto en M5 Pro, batch 16)

¿Empezar la traducción? [s/N]:
```

Para dejarlo corriendo en segundo plano usa `-q`, que empieza sin preguntar (sin `-q` el script
se detiene, porque no hay dónde mostrar la pregunta):

```bash
python -u translate_epub.py libro.epub -q > translation.log 2>&1 &        # venv activado
uv run python -u translate_epub.py libro.epub -q > translation.log 2>&1 & # uv
tail -f translation.log
```

`-u` hace que el progreso se escriba en el log al instante, en vez de acumularse.

### 2. Armar el EPUB

```bash
python build_epub.py libro.epub
```

- Si la traducción terminó, genera `libro (ES).epub`.
- Si no, genera `libro (ES parcial).epub` con los capítulos traducidos hasta el primero
  incompleto. El índice se ajusta y el título lleva "(parcial)".

Se puede ejecutar mientras `translate_epub.py` sigue traduciendo.

### Retomar o corregir

Vuelve a ejecutar el mismo comando de traducción: solo traduce lo que falta. Las traducciones
defectuosas de ejecuciones anteriores (bucles, respuestas que no son traducciones) se detectan
y se rehacen solas.

### Opciones

| Opción | Script | Descripción |
|---|---|---|
| `--src de` / `--tgt es` | ambos | Idiomas origen y destino (códigos ISO, p. ej. `en`, `fr`, `pt-BR`) |
| `--batch 16` | `translate_epub.py` | Párrafos en paralelo. 32 rinde ~10% más y usa el doble de memoria; bájalo si tu Mac tiene poca RAM |
| `-q`, `--quiet` | `translate_epub.py` | Empezar sin mostrar el plan ni pedir confirmación |
| `salida.epub` | `build_epub.py` | Ruta del EPUB generado (por defecto, junto al original) |

Si cambias los idiomas, usa los mismos `--src` y `--tgt` en los dos scripts.

### Archivos de trabajo

Todo queda en `work/<nombre del libro>/`:

| Ruta | Contenido |
|---|---|
| `src/` | El EPUB original extraído (no se modifica) |
| `cache_<src>-<tgt>.jsonl` | Las traducciones: es el progreso. Si lo borras, se empieza de cero |
| `build/` | El contenido del último EPUB armado |

## Google Colab

[`epub_translator_colab.ipynb`](epub_translator_colab.ipynb) hace lo mismo en una GPU NVIDIA de
Google Colab, sin instalar nada en tu computadora. Usa el mismo método de traducción (párrafos,
formato, caché retomable y EPUB parcial), pero con el modelo original de Google mediante
[transformers](https://huggingface.co/docs/transformers) en vez de MLX.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papatr0x/epub-translator/blob/main/epub_translator_colab.ipynb)

Antes de usarlo:

1. **GPU:** en Colab, *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*
   (o L4/A100 con Colab Pro).
2. **Licencia:** el modelo [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)
   requiere iniciar sesión en Hugging Face y aceptar los términos de Gemma.
3. **Token:** crea un token de lectura en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   y guárdalo en Colab, en el panel 🔑 *Secretos*, con el nombre `HF_TOKEN`. Si el token es
   *fine-grained*, activa el acceso de lectura a los repos con licencia ("gated").
4. **El libro:** súbelo a tu Google Drive, o deja que el notebook te pida subirlo.

Luego ejecuta las celdas en orden. La caché se guarda en Google Drive
(`MyDrive/epub-translator/work/`): si Colab se desconecta, vuelve a ejecutar las celdas y sigue
donde quedó. La celda 7 arma el EPUB, completo o parcial, en cualquier momento.

Notas:
- La T4 gratuita no soporta bf16 y el notebook usa fp16, con el que Gemma a veces produce
  salidas vacías o sin sentido. La celda 5 hace una prueba y avisa si pasa; en ese caso usa una
  GPU L4 o A100.
- El notebook no usa `requirements.txt`: instala sus dependencias en la celda 2.

## Cómo funciona

1. **Unidades:** cada párrafo, título o celda es una unidad. Traducir el párrafo completo le
   da contexto al modelo, y reemplazar solo su texto conserva el HTML original.
2. **Formato:** la cursiva y la negrita viajan como Markdown (`*x*`, `**x**`), que el modelo
   respeta mucho mejor que las etiquetas HTML. Al volver, el Markdown se convierte otra vez en
   las etiquetas originales, aunque el modelo haya cambiado algún marcador: al libro nunca
   llegan asteriscos. Los párrafos con links o notas al pie se traducen por fragmentos para
   no perderlos.
3. **Validación:** si una salida entra en bucle, se corta o es una "explicación" en vez de una
   traducción, se reintenta con penalización de repetición. Si vuelve a fallar, se conserva el
   texto original y queda anotado en el log.
4. **Caché:** cada traducción se agrega a un archivo JSONL apenas termina. La clave es un hash
   de modelo + idiomas + texto: lo repetido se traduce una sola vez.

El detalle de las optimizaciones (por qué es rápido, cómo retoma) está en el docstring de
[`translate_epub.py`](translate_epub.py). Las convenciones de código están en
[`code-style.md`](code-style.md).

## Limitaciones

- Es traducción automática: sirve para lectura personal, no tiene calidad editorial. Los
  coloquialismos y juegos de palabras suelen salir literales.
- A veces el modelo quita una cursiva al traducir. Se recupera si su texto queda igual (nombres,
  títulos); si no, se pierde ese formato, nunca el texto.
- Los fragmentos muy cortos y sin contexto que el modelo no logra traducir (letras decorativas,
  un "th" en superíndice) se dejan en el idioma original.
- La estimación de tiempo usa como referencia un M5 Pro; en otras máquinas varía.

## Licencias

Los modelos TranslateGemma están sujetos a los
[términos de uso de Gemma](https://ai.google.dev/gemma/terms). Traduce solo libros que tengas
derecho a usar.
