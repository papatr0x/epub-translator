# epub-translator

Traduce libros EPUB en tu Mac con Apple Silicon usando
[Gemma 4](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit) (o TranslateGemma) sobre
[MLX](https://github.com/ml-explore/mlx). Es gratis y todo corre localmente: el libro no se
envía a ningún servicio.

- **Conserva el libro:** traduce párrafo por párrafo y mantiene la estructura, la cursiva y
  la negrita, los links, las notas al pie y el índice.
- **Rápido:** traduce varios párrafos en paralelo. En un M5 Pro, una novela de ~3000 párrafos
  tarda unos 27 minutos con el modelo por defecto (11 con el más rápido).
- **Retomable:** cada párrafo traducido se guarda al instante. Si se corta (Ctrl+C, un cierre
  inesperado), al relanzar sigue donde quedó.
- **EPUB parcial:** puedes armar un EPUB con los capítulos ya traducidos mientras la
  traducción sigue corriendo.

¿No tienes un Mac con Apple Silicon? Usa el [notebook para Google Colab](#google-colab).

## Requisitos

- Mac con Apple Silicon (M1 o posterior). MLX no funciona en Intel, Linux ni Windows.
- Python 3.10 o posterior (probado con 3.14), o [uv](https://docs.astral.sh/uv/).
- Memoria libre según el [modelo](#modelos): ~24 GB con el modelo por defecto (Gemma 4), ~7 GB con `--modelo tg4b`.
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

La primera traducción descarga el modelo de Hugging Face (~18 GB el de por defecto; ver
[Modelos](#modelos)) a `~/.cache/huggingface/`.
No hace falta cuenta. Las siguientes ejecuciones lo usan desde ahí.

## Uso

Los ejemplos usan `python` con el entorno **activado** (opción A). Con **uv** (opción B),
reemplaza `python` por `uv run`:

| venv activado | uv |
|---|---|
| `python translate_epub.py libro.epub --src de` | `uv run translate_epub.py libro.epub --src de` |
| `python build_epub.py libro.epub --src de` | `uv run build_epub.py libro.epub --src de` |

Sin activar el entorno también funciona con la ruta completa: `.venv/bin/python translate_epub.py …`.

El proceso tiene dos pasos independientes: **traducir** (lento, llena una caché) y **armar**
el EPUB (instantáneo, lee la caché). Puedes armar el EPUB tantas veces como quieras.

### 1. Traducir

```bash
python translate_epub.py libro.epub --src de
```

Antes de cargar el modelo muestra qué va a hacer y pregunta si empezar:

```
Plan de traducción
  Libro:      libro.epub (94 archivos HTML)
  Idiomas:    de → es
  Modelo:     mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit (batch 32)
  Unidades:   3035 en total, 0 ya traducidas, 3035 pendientes
  A traducir: 3050 textos, ~27 min estimados (referencia: 0.54 s/texto en M5 Pro, batch 32)

¿Empezar la traducción? [s/N]:
```

Para dejarlo corriendo en segundo plano usa `-q`, que empieza sin preguntar (sin `-q` el script
se detiene, porque no hay dónde mostrar la pregunta):

```bash
python -u translate_epub.py libro.epub --src de -q > translation.log 2>&1 &        # venv activado
uv run python -u translate_epub.py libro.epub --src de -q > translation.log 2>&1 & # uv
tail -f translation.log
```

`-u` hace que el progreso se escriba en el log al instante, en vez de acumularse.

### 2. Armar el EPUB

```bash
python build_epub.py libro.epub --src de
```

- Si la traducción terminó, genera `libro (ES).epub`.
- Si no, genera `libro (ES parcial).epub` con los capítulos traducidos hasta el primero
  incompleto. El índice se ajusta y el título lleva "(parcial)".

Se puede ejecutar mientras `translate_epub.py` sigue traduciendo.

### Retomar o corregir

Vuelve a ejecutar el mismo comando de traducción: solo traduce lo que falta. Las traducciones
defectuosas de ejecuciones anteriores (bucles, respuestas que no son traducciones) se detectan
y se rehacen solas.

### Notas de traducción (acepciones y modismos)

Algunas palabras tienen varias acepciones y el modelo a veces elige la equivocada. Por
ejemplo, el alemán *knipsen* (sacar fotos) a veces se traducía como "golpear" o "jugar". Puedes
explicarle al modelo qué acepción usar escribiendo notas en `notas/<src>-<tgt>.md` (por ejemplo, `notas/de-es.md`).
Cada usuario crea las suyas: la carpeta `notas/` no está en el repositorio (`.gitignore`).

```markdown
## knips
«knipsen» y sus formas (knipste, geknipst, Knipsen) significan siempre sacar fotos:
tradúcelo como «fotografiar» o «sacar/tomar fotos». Nunca como golpear, jugar ni mirar.
```

- Cada nota empieza con `## <patrón>`: una expresión regular, sin distinguir mayúsculas, que
  se busca en el texto original (`knips` también encuentra `geknipst` o `Meisterknipser`).
  Debajo va la explicación en lenguaje natural, como se la darías a un traductor.
- La nota se agrega al prompt **solo** de los párrafos donde aparece el patrón.
- Las notas de `notas/` se aplican a todos los libros de ese par de idiomas. Para notas de un solo libro (nombres, jerga propia), crea `libro.notas.md` junto
  al EPUB, con el mismo formato.
- Si agregas o cambias una nota, al relanzar la traducción se rehacen **solo** los párrafos
  que la usan. El plan muestra cuántos son.

Con la nota de *knipsen*, los 37 párrafos del libro de prueba que usan esa palabra quedaron
con la acepción correcta, contra 2 de cada 10 sin la nota.

### Opciones

| Opción | Script | Descripción |
|---|---|---|
| `--src de` | ambos | **Obligatorio.** Idioma del libro (código ISO, p. ej. `de`, `en`, `fr`, `pt-BR`) |
| `--tgt es` | ambos | Idioma destino. Por defecto, español |
| `--batch 32` | `translate_epub.py` | Párrafos en paralelo. 32 es el valor más rápido medido; bájalo si tu Mac tiene poca RAM (si la GPU se queda sin memoria, el script sigue solo con la mitad) |
| `-q`, `--quiet` | `translate_epub.py` | Empezar sin mostrar el plan ni pedir confirmación |
| `--modelo gemma4` | ambos | Modelo a usar (ver [Modelos](#modelos)). Por defecto, el último usado con ese libro |
| `salida.epub` | `build_epub.py` | Ruta del EPUB generado (por defecto, junto al original) |

Usa los mismos `--src` y `--tgt` en los dos scripts: los idiomas son parte de la caché.

### Modelos

| `--modelo` | Modelo | Memoria | Velocidad* | Calidad |
|---|---|---|---|---|
| `tg4b` | [TranslateGemma 4B](https://huggingface.co/mlx-community/translategemma-4b-it-4bit) | ~7 GB | 0.2 s/párrafo | Buena en frases simples; falla en modismos y acepciones raras |
| `tg12b` | [TranslateGemma 12B](https://huggingface.co/mlx-community/translategemma-12b-it-4bit) | ~20 GB | 0.7 s/párrafo | Mejor vocabulario |
| `gemma4` (por defecto) | [Gemma 4 26B-A4B](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit) | ~24 GB | 0.5 s/párrafo | La mejor: traduce modismos por su sentido |

\* Memoria pico y velocidad medidas en un M5 Pro con lote 32. Un libro de ~3000 párrafos:
~11 min (`tg4b`), ~33 min (`tg12b`), ~27 min (`gemma4`). Si tu Mac no tiene memoria para
Gemma 4, usa `--modelo tg4b`.

En una prueba con 32 párrafos difíciles del libro de prueba (palabras con varias acepciones,
jerga de la RDA), sin notas, acertaron 15 (`tg4b`), 21 (`tg12b`) y 30 (`gemma4`). Por ejemplo,
*klotzte ran* («trabajaba sin parar»): `tg4b` «intervino de forma abrupta», `gemma4` «no
escatimaba esfuerzos».

Gemma 4 es un modelo general, no uno especializado en traducción: recibe instrucciones de
traductor literario, y a veces ajusta detalles de estilo (por ejemplo, rayas de diálogo en vez
de comillas). Las [notas de traducción](#notas-de-traducción-acepciones-y-modismos) funcionan
con todos los modelos.

El modelo forma parte de la caché: cambiarlo retraduce el libro, y las traducciones del modelo
anterior quedan guardadas. Cada libro recuerda el último modelo usado, así que `build_epub.py`
y las siguientes ejecuciones lo usan sin repetir `--modelo`.

### Archivos de trabajo

Todo queda en `work/<nombre del libro>/`:

| Ruta | Contenido |
|---|---|
| `src/` | El EPUB original extraído (no se modifica) |
| `cache_<src>-<tgt>.jsonl` | Las traducciones: es el progreso. Si lo borras, se empieza de cero |
| `build/` | El contenido del último EPUB armado |
| `modelo.txt` | El último modelo usado con este libro |

## Google Colab

[`epub_translator_colab.ipynb`](epub_translator_colab.ipynb) hace lo mismo en una GPU NVIDIA de
Google Colab, sin instalar nada en tu computadora. Usa el mismo método de traducción (párrafos,
formato, caché retomable y EPUB parcial), pero con el modelo original de Google mediante
[transformers](https://huggingface.co/docs/transformers) en vez de MLX.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papatr0x/epub-translator/blob/main/epub_translator_colab.ipynb)

Antes de usarlo:

1. **GPU:** en Colab, *Entorno de ejecución → Cambiar tipo de entorno de ejecución* → L4 o
   A100 (Colab Pro) para el modelo por defecto, Gemma 4. Con la T4 gratuita elige
   translategemma-4b-it en la celda 1.
2. **Modelo y licencia:** elige el modelo en la celda 1:
   [translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it),
   [translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it) o
   [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). Entra a su página
   con tu cuenta de Hugging Face y acepta los términos si los pide. Si el modelo no entra
   completo en la GPU, se carga cuantizado a 4 bits: en una **L4 (24 GB)**, TranslateGemma
   12B y Gemma 4 van en 4 bits (8 bits solo si lo eliges: genera mucho más lento).
3. **Token:** crea un token de lectura en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   y guárdalo en Colab, en el panel 🔑 *Secretos*, con el nombre `HF_TOKEN`. Si el token es
   *fine-grained*, activa el acceso de lectura a los repos con licencia ("gated").
4. **El libro:** súbelo a tu Google Drive, o deja que el notebook te pida subirlo.

Las [notas de traducción](#notas-de-traducción-acepciones-y-modismos) también funcionan: el
notebook las busca en `MyDrive/epub-translator/notas/<src>-<tgt>.md` (configurable en la
celda 1) y en `libro.notas.md` junto al EPUB. Puedes copiar ahí las de tu carpeta `notas/`.

En la celda 1 indica el idioma del libro (`SRC_LANG`, obligatorio; el destino es español salvo
que cambies `TGT_LANG`). Luego ejecuta las celdas en orden. La caché se guarda en Google Drive
(`MyDrive/epub-translator/work/`): si Colab se desconecta, vuelve a ejecutar las celdas y sigue
donde quedó. La celda 7 arma el EPUB, completo o parcial, en cualquier momento.

Notas:
- La T4 gratuita no soporta bf16 y el notebook usa fp16, con el que Gemma a veces produce
  salidas vacías o sin sentido. La celda 5 hace una prueba y avisa si pasa; en ese caso usa una
  GPU L4 o A100.
- Las comparaciones de calidad y velocidad de los modelos se hicieron en un Mac con MLX. En
  Colab, la cuantización de 8 o 4 bits (bitsandbytes) es distinta y transformers es más lento
  con Gemma 4: la calidad debería ser parecida, pero la velocidad no está medida.
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
   traducción, se reintenta con penalización de repetición y el doble de margen. Si vuelve a
   fallar, se conserva el texto original, queda anotado en el log y se reintenta la próxima
   vez que lances la traducción; `build_epub.py` avisa cuántos textos quedaron así.
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
