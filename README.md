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

## Inicio rápido

```bash
git clone https://github.com/papatr0x/epub-translator.git
cd epub-translator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python translate_epub.py libro.epub   # muestra el plan y pide confirmación
.venv/bin/python build_epub.py libro.epub       # genera "libro (ES).epub"
```

La primera vez se descarga el modelo de Hugging Face (~2 GB), sin necesidad de cuenta.

## Requisitos

- Mac con Apple Silicon (M1 o posterior). MLX no funciona en Intel, Linux ni Windows.
- Python 3.10 o posterior (probado con 3.14).
- Unos 3 GB de memoria libre (medido con el lote por defecto, 16).
- El libro en formato **EPUB**. Un PDF hay que convertirlo antes, por ejemplo con
  [Calibre](https://calibre-ebook.com): `ebook-convert libro.pdf libro.epub`.

## Uso

El proceso tiene dos pasos independientes: **traducir** (lento, llena una caché) y **armar**
el EPUB (instantáneo, lee la caché). Puedes armar el EPUB tantas veces como quieras.

### 1. Traducir

```bash
.venv/bin/python translate_epub.py libro.epub
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
.venv/bin/python -u translate_epub.py libro.epub -q > translation.log 2>&1 &
tail -f translation.log
```

### 2. Armar el EPUB

```bash
.venv/bin/python build_epub.py libro.epub
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

## Cómo funciona

1. **Unidades:** cada párrafo, título o celda es una unidad. Traducir el párrafo completo le
   da contexto al modelo, y reemplazar solo su texto conserva el HTML original.
2. **Formato:** la cursiva y la negrita viajan como Markdown (`*x*`, `**x**`), que el modelo
   respeta mucho mejor que las etiquetas HTML. Si lo rompe, ese párrafo se traduce como texto
   plano. Los párrafos con links o notas al pie se traducen por fragmentos para no perderlos.
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
- Si el modelo no respeta la cursiva de un párrafo, ese párrafo pierde el formato, nunca el texto.
- Los fragmentos muy cortos y sin contexto que el modelo no logra traducir (letras decorativas,
  un "th" en superíndice) se dejan en el idioma original.
- La estimación de tiempo usa como referencia un M5 Pro; en otras máquinas varía.

## Licencias

Los modelos TranslateGemma están sujetos a los
[términos de uso de Gemma](https://ai.google.dev/gemma/terms). Traduce solo libros que tengas
derecho a usar.
