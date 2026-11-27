#!/usr/bin/env python3
"""
Traduce un EPUB con un modelo local: Gemma 4 o TranslateGemma (MLX, Apple Silicon, gratis).

Características:
- Traduce párrafo por párrafo (bloque), conservando la estructura y el formato
  inline (cursiva/negrita vía Markdown; links y notas fragmento a fragmento).
- Batching continuo: traduce varios párrafos en paralelo en la GPU.
- Resume a nivel de párrafo: cada traducción se guarda al instante en una caché
  (JSONL). Si se interrumpe, al relanzar solo se traduce lo que falta.
- Traduce en orden de lectura (spine), así build_epub.py puede armar un EPUB
  parcial con los capítulos ya traducidos mientras esto sigue corriendo.
- Directorio de trabajo por libro: ./work/<nombre_epub>/

Este script solo extrae y traduce (a la caché). El EPUB lo arma build_epub.py.

Diseño y optimizaciones (por qué es rápido y retomable)
───────────────────────────────────────────────────────
La generación de texto con un LLM en Apple Silicon está limitada por el ancho de
banda de memoria, no por el cómputo: para producir CADA token hay que leer
todos los pesos del modelo desde la memoria. De ahí salen las tres mejoras
de velocidad más grandes:

1. Token de parada <end_of_turn>  (Translator._load)
   El tokenizer de Gemma solo declara <eos> como fin, pero el modelo cierra su
   respuesta con <end_of_turn>. Sin agregarlo como token de parada el modelo
   seguía generando basura hasta max_tokens: la mayor parte del tiempo se iba
   en tokens que después se descartaban (~10 s por párrafo largo).

2. Modelo cuantizado a 4 bits  (MODEL_NAME)
   Si cada token exige leer todos los pesos, leer la mitad de bytes (4 vs 8
   bits) da ~1.5-2x más tokens/s. En un modelo de 4B la pérdida de calidad
   es pequeña para traducción.

3. Batching continuo  (Translator.translate_many)
   Con una sola secuencia, la GPU pasa la mayor parte del tiempo esperando a
   la memoria. Con N secuencias en paralelo, los pesos leídos en cada paso
   sirven para producir N tokens a la vez: el costo por paso casi no crece y
   el rendimiento se multiplica. Es "continuo" porque cuando una secuencia
   termina entra otra al instante; un batch estático esperaría al párrafo más
   largo del grupo, y acá los largos varían mucho (de "1" a 500 palabras).
   Medido en un M5 Pro con TranslateGemma 4B: 1.30 s/párrafo secuencial →
   0.55 con 16 secuencias → 0.31 con 32 (cada paso tarda lo mismo con 16 que
   con 32). Con más de 32 cada paso se encarece y deja de rendir; en Gemma 4,
   que activa distintos "expertos" por secuencia, la ganancia es menor
   (0.77 → 0.70).

Otras decisiones:

4. max_tokens dinámico  (Translator._prompt)
   Proporcional al largo de la entrada (×2.5 + 64). Una traducción normal
   alemán→español rara vez llega a 2x el largo del original, así que no se
   corta; y si el modelo entra en un bucle, el costo queda acotado.

5. Caché JSONL append-only con clave hash  (TranslationCache)
   - Cada traducción se escribe y se hace flush al terminar: si el proceso
     muere se pierde como mucho lo que estaba generándose en ese momento.
   - La clave es sha1(modelo + idiomas + texto): textos repetidos (números
     de capítulo, títulos) se traducen una sola vez, y cambiar de modelo o de
     idioma no mezcla resultados.
   - Una línea truncada por un corte se ignora al leer y se cierra con "\\n"
     antes de seguir escribiendo.

6. Planificar = aplicar  (run + apply_translations)
   Para saber qué falta traducir se ejecuta el MISMO código que aplica las
   traducciones, con un `lookup` que anota los textos ausentes de la caché.
   Así las claves que se piden son exactamente las que después se buscan, sin
   llevar un registro aparte que pueda desincronizarse. Parsear HTML cuesta
   milisegundos; traducir, segundos: recalcular el plan es gratis.

7. Orden de lectura  (spine_files)
   Se traduce en el orden del spine, con los títulos del índice primero. Así
   cada capítulo queda completo apenas se traduce, y build_epub.py puede armar
   un EPUB parcial en cualquier momento. Cada párrafo necesita una sola
   traducción: aunque el modelo altere el Markdown, la salida se aprovecha.

8. Unidad = párrafo, formato como Markdown  (sección "Unidades de traducción")
   Traducir el párrafo entero da contexto al modelo y conserva la estructura.
   Las etiquetas inline se pasan como Markdown (*x*, **x**) porque el modelo
   lo vio mucho en su entrenamiento y lo respeta; las etiquetas tipo XML
   (<g1>...</g1>) las descartaba casi siempre. Al volver, el Markdown se
   convierte en HTML de forma tolerante (md_to_nodes): un marcador cambiado
   no tira el formato de todo el párrafo, y ningún asterisco llega al libro.

9. Validación y reintento  (output_problem, translate_many)
   La decodificación greedy a veces cae en bucles ("die die die...") o, ante
   fragmentos sin contexto, el modelo "explica" en vez de traducir. Se detecta
   por heurística y se reintenta una vez con penalización de repetición. La
   penalización solo se usa en el reintento porque en general también castiga
   repeticiones legítimas (nombres propios, estribillos). El reintento tiene
   además el doble de max_tokens, por si la salida se cortó siendo legítima.
   Si también falla, se guarda el original marcado como fallo: el EPUB se
   puede armar igual y la próxima ejecución lo vuelve a intentar.

10. Recorrido O(n) del HTML  (collect_units)
    Un solo recorrido recursivo decide qué bloques son "hoja" en vez de
    reescanear los descendientes de cada bloque (O(n²) en HTML anidado).

11. mlx_lm se importa solo dentro de Translator
    build_epub.py reutiliza este módulo sin cargar el modelo ni la GPU.

12. Notas de traducción  (sección "Notas de traducción")
    Explicaciones en lenguaje natural sobre qué acepción usar, que se insertan
    en el prompt solo de los textos donde aparece su patrón. Forman parte de la
    clave de caché de esos textos: cambiar una nota retraduce solo lo afectado,
    y los textos sin notas conservan la clave de siempre.

13. Modelos elegibles  (MODELS, --modelo)
    TranslateGemma 4B/12B usan su propia plantilla; Gemma 4 26B-A4B es un modelo
    general que recibe instrucciones de traductor literario (CHAT_SYSTEM_PROMPT)
    con el razonamiento desactivado. En 52 párrafos del libro de prueba, Gemma 4
    acertó la acepción en 30 de 32 casos difíciles (TG 12B: 21, TG 4B: 15) y
    tradujo los modismos por su sentido. Es más rápido que TG 12B pese a ser más
    grande porque es "mixture of experts": usa ~4B parámetros por token. Por
    eso es el modelo por defecto. El modelo usado se recuerda por libro (work/<libro>/modelo.txt) para que
    build_epub.py busque en la caché las traducciones de ese mismo modelo.

14. Menos trabajo al procesar los prompts  (Translator.translate_many)
    Con el batching ya afinado, procesar los prompts (prefill) era el 20% del
    tiempo. Tres cambios que juntos lo bajan a la mitad y, de paso, abaratan
    la generación. Medido con Gemma 4 en 300 párrafos: 201 s → 160 s (-21%;
    con TranslateGemma 4B, -30%);
    cada uno por separado rinde poco (-3% a -10%), se potencian entre sí.
    - Caché del prefijo: todos los prompts empiezan igual (las instrucciones de
      traductor, ~90 tokens de los ~200 de un prompt típico). Ese comienzo se
      procesa una sola vez y cada texto parte de una copia de su resultado (el
      KV cache). Un prompt con notas de traducción empieza distinto y se
      procesa entero, como antes.
    - Admisión en grupos: cada pasada del modelo cuesta casi lo mismo con 1
      prompt que con 4 (lo que cuesta es leer los pesos). En vez de procesar
      un prompt cada vez que termina un párrafo, se espera a tener 4 lugares
      libres y se procesan juntos.
    - Orden por largo en ventanas: dentro de un batch, las secuencias se
      rellenan hasta el largo de la mayor. Agrupar textos de largo parecido
      reduce ese relleno. Se ordena dentro de ventanas de 8 lotes (256 textos)
      y no el libro entero, para que los capítulos se sigan completando de
      principio a fin (punto 7): un EPUB parcial se atrasa como mucho una
      ventana.
    Cambiar qué textos comparten batch cambia levemente los cálculos, y con
    eso algunas palabras de la traducción (la calidad es la misma). Lo ya
    guardado en la caché no se toca.

Uso:
    .venv/bin/python translate_epub.py <entrada.epub> --src <idioma> [--tgt es] [--batch 32] [-q]

Antes de empezar muestra el plan (cuánto falta, tiempo estimado) y pide
confirmación. -q empieza sin preguntar; es obligatorio en segundo plano.

Ejemplos:
    .venv/bin/python translate_epub.py input.epub --src de
    .venv/bin/python -u translate_epub.py input.epub --src de -q > translation.log 2>&1 &
"""

import argparse
import copy
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import sys
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

# ─── Config ────────────────────────────────────────────────────────────────────

class ModelSpec(NamedTuple):
    repo: str              # repositorio MLX en Hugging Face
    kind: str              # "translategemma" (plantilla propia) o "chat" (modelo general)
    ref_seconds: float     # s/texto medidos en un M5 Pro con batch 32 (para estimar)
    summary: str


# Modelos elegibles con --modelo (docstring, punto 13). Todos en 4 bits: la mitad
# de bytes a leer por token que en 8 bits (punto 2). Calidad medida en 52 párrafos
# del libro de prueba; el modelo es parte de la clave de caché, así que cambiarlo
# retraduce el libro.
MODELS = {
    "tg4b": ModelSpec("mlx-community/translategemma-4b-it-4bit", "translategemma", 0.22,
                      "TranslateGemma 4B: el más rápido, ~7 GB de memoria"),
    "tg12b": ModelSpec("mlx-community/translategemma-12b-it-4bit", "translategemma", 0.66,
                       "TranslateGemma 12B: mejor vocabulario, ~20 GB"),
    "gemma4": ModelSpec("mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit", "chat", 0.54,
                        "Gemma 4 26B-A4B: la mejor calidad (modismos, acepciones), ~24 GB"),
}
# Gemma 4 por defecto: es el que mejor traduce modismos y expresiones (punto 13)
DEFAULT_MODEL = "gemma4"
# Modelo activo; lo cambia select_model()
MODEL_NAME = MODELS[DEFAULT_MODEL].repo
MODEL_KIND = MODELS[DEFAULT_MODEL].kind
# El idioma origen se indica siempre (--src); el destino es español salvo --tgt
DEFAULT_TARGET_LANG = "es"

# max_tokens de salida = tokens de entrada * factor + margen (docstring, punto 4)
MAX_TOKENS_FACTOR = 2.5
MAX_TOKENS_MARGIN = 64
# Secuencias generadas en paralelo (docstring, punto 3). 32 es el mejor valor
# medido en los tres modelos: con 48 o más cada paso tarda más y el total empeora.
# El KV cache (memoria que guarda el contexto de cada secuencia activa) crece con
# el número de secuencias; si la GPU se queda sin memoria, translate_many sigue
# con la mitad.
BATCH_SIZE = 32
# Orden de traducción (docstring, punto 14): ventanas de WINDOW_BATCHES lotes en
# orden de lectura, y dentro de cada ventana de mayor a menor largo.
WINDOW_BATCHES = 8
# Los textos nuevos entran al batch de a ADMIT_GROUP (docstring, punto 14).
ADMIT_GROUP = 4
PROGRESS_EVERY_S = 30
# Velocidad de referencia para estimar la duración antes de empezar (medida en un
# M5 Pro con batch 32). Es solo orientativa: depende de la máquina y del batch.
REF_SECONDS_PER_TEXT = MODELS[DEFAULT_MODEL].ref_seconds
# Reintento ante salidas malas (docstring, punto 9). La penalización divide los
# logits de los tokens generados en los últimos RETRY_REPETITION_CONTEXT pasos,
# así que repetir lo mismo pierde probabilidad y el bucle se rompe. 1.2 corrigió
# el bucle real del libro; 1.1 todavía dejaba restos.
RETRY_REPETITION_PENALTY = 1.2
RETRY_REPETITION_CONTEXT = 64
# Salida sospechosa si supera len(origen) * factor + margen (el modelo "explica"
# en vez de traducir, típico en fragmentos cortos sin contexto)
MAX_OUTPUT_RATIO = 3
MAX_OUTPUT_MARGIN = 50
END_OF_TURN = "<end_of_turn>"
# Marcas de fin de turno y de "razonamiento" que pueden quedar en la salida
# (Gemma 3 usa <end_of_turn>; Gemma 4, <turn|> y un canal <|channel>...<channel|>)
END_MARKERS_RE = re.compile(r"<end_of_turn>|<turn\|>")
THINKING_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)

# Instrucciones para modelos generales ("chat"). TranslateGemma no las necesita:
# su plantilla ya fija el rol de traductor.
CHAT_SYSTEM_PROMPT = (
    "You are a professional literary translator from {src} to {tgt}. Translate the "
    "user's text into {tgt}. Preserve meaning, tone and register; translate colloquialisms "
    "and idioms by their meaning, not word for word. Keep the Markdown emphasis markers "
    "(*x*, **x**) around the corresponding words, and keep the original quotation marks "
    "and punctuation style. Reply only with the translation, without comments, notes or "
    "surrounding quotes.")
LANG_NAMES = {"de": "German", "en": "English", "es": "Spanish", "fr": "French",
              "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
              "ru": "Russian", "ca": "Catalan", "sv": "Swedish", "da": "Danish"}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code.split("-")[0].split("_")[0].lower(), code)


def select_model(name: str) -> str:
    """Activa un modelo por alias (tg4b, tg12b, gemma4) o repositorio MLX.
    Un repositorio desconocido se trata como "chat" salvo que su nombre diga
    translategemma. Devuelve el repositorio."""
    global MODEL_NAME, MODEL_KIND, REF_SECONDS_PER_TEXT
    spec = MODELS.get(name) or next((m for m in MODELS.values() if m.repo == name), None)
    if spec is None:
        spec = ModelSpec(name, "translategemma" if "translategemma" in name.lower() else "chat",
                         MODELS["gemma4"].ref_seconds, name)
    MODEL_NAME, MODEL_KIND, REF_SECONDS_PER_TEXT = spec.repo, spec.kind, spec.ref_seconds
    return spec.repo


def is_oom(e: Exception) -> bool:
    """¿El error es falta de memoria de la GPU (Metal)?"""
    msg = str(e).lower()
    return any(k in msg for k in ("insufficient memory", "out of memory", "unable to allocate"))


def clean_output(text: str) -> str:
    """Quita marcas de control de la salida del modelo."""
    return END_MARKERS_RE.split(THINKING_RE.sub("", text))[0].strip()

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR / "work"

HTML_EXTENSIONS = {".html", ".xhtml", ".htm"}
SKIP_TAGS = {"style", "script", "head", "title", "link", "meta", "svg", "math"}
BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "td", "th",
    "caption", "blockquote", "div", "figcaption", "pre", "section", "article",
    "aside", "header", "footer", "nav", "ul", "ol", "dl", "table", "tr",
    "tbody", "thead", "tfoot", "figure", "body", "html",
}
INLINE_TAGS = {
    "a", "abbr", "b", "bdi", "bdo", "cite", "code", "dfn", "em", "i", "kbd",
    "mark", "q", "s", "samp", "small", "span", "strong", "sub", "sup", "time",
    "u", "var", "font", "del", "ins",
}

# lookup(texto) -> traducción o None. Es la única forma en que el código de HTML
# accede a las traducciones: según qué lookup se le pase, el mismo código sirve
# para planificar (anotar lo que falta) o para aplicar (docstring, punto 6).
Lookup = Callable[[str], "str | None"]

# Inline que se traduce como Markdown (y se reconstruye)
MD_TAGS = {"em": "*", "i": "*", "cite": "*", "dfn": "*", "var": "*",
           "b": "**", "strong": "**"}
# Inline que obliga a traducir fragmento por fragmento para no perderlo
# (links, notas al pie, saltos de línea e imágenes no se pueden expresar en
# Markdown de forma que el modelo los respete)
PRESERVE_TAGS = {"a", "sup", "sub", "br", "img", "wbr", "code", "kbd", "samp"}
# Elementos sin texto que igual se ven: no son "anclas" que se puedan mover
VISIBLE_EMPTY_TAGS = ["br", "img", "wbr", "hr", "svg", "image", "math"]


# ─── Notas de traducción (acepciones, modismos) ────────────────────────────────
#
# Archivos Markdown donde cada nota empieza con "## <patrón>" y sigue con la
# explicación en lenguaje natural:
#
#     ## knips
#     «knipsen» y sus formas significan siempre sacar fotos: tradúcelo como
#     «fotografiar» o «sacar/tomar fotos». Nunca como golpear, jugar ni mirar.
#
# El patrón es una expresión regular (sin distinguir mayúsculas) que se busca en
# el texto ORIGINAL. Cada nota se agrega al prompt solo de los textos donde
# aparece su patrón: el prompt queda corto y el resto del libro no cambia.
#
# Por qué funciona: TranslateGemma sigue instrucciones de terminología puestas
# en su prompt. Probado con "knipsen" en 10 párrafos mal traducidos: sin nota, 2
# usaban la acepción correcta; con la nota, 10, sin que el texto de la nota se
# colara en la traducción.
#
# Se combinan dos archivos:
# - notas/<src>-<tgt>.md: notas de cada usuario (fuera de git), comunes a todos
#   los libros de ese par de idiomas (modismos, usos que se repiten);
# - <libro>.notas.md junto al EPUB: notas propias de un libro (nombres, jerga).

NOTES_DIR = SCRIPT_DIR / "notas"
# El encabezado del bloque de notas va en el idioma destino cuando se conoce
# (probado: en español da resultados algo mejores que en inglés para de → es).
NOTES_HEADER = {"es": "Terminología (respétala siempre):"}
NOTES_HEADER_DEFAULT = "Terminology (follow strictly):"
# Las notas se insertan en el prompt de TranslateGemma justo antes de esta frase
# de su plantilla, es decir, después de su descripción del rol de traductor.
PROMPT_NOTES_ANCHOR = "Produce only the"


class Note(NamedTuple):
    pattern: re.Pattern
    text: str
    origin: str  # "archivo:línea", para mensajes


def parse_notes(path: Path) -> list[Note]:
    """Lee un archivo de notas. El texto antes del primer "## " se ignora
    (sirve para explicar el archivo)."""
    notes: list[Note] = []
    current: tuple[re.Pattern, list[str], str] | None = None

    def close():
        if current:
            body = " ".join(" ".join(current[1]).split())
            if body:
                notes.append(Note(current[0], body, current[2]))

    # Los comentarios <!-- --> se ignoran (detect_terms.py los usa para dejar
    # estadísticas y ejemplos); se reemplazan por saltos de línea para que los
    # números de línea de los mensajes sigan coincidiendo con el archivo.
    raw_text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"),
                      path.read_text(encoding="utf-8"), flags=re.DOTALL)
    for n, line in enumerate(raw_text.splitlines(), 1):
        if line.startswith("## "):
            close()
            raw = line[3:].strip()
            try:
                current = (re.compile(raw, re.IGNORECASE), [], f"{path.name}:{n}")
            except re.error as e:
                raise SystemExit(f"❌ {path}:{n}: patrón inválido {raw!r}: {e}")
        elif current is not None:
            current[1].append(line)
    close()
    return notes


def notes_files(input_epub: Path, src_lang: str, tgt_lang: str) -> list[Path]:
    """Archivos de notas que existen para este libro y par de idiomas."""
    candidates = [NOTES_DIR / f"{src_lang}-{tgt_lang}.md",
                  input_epub.with_name(f"{input_epub.stem}.notas.md")]
    return [p for p in candidates if p.exists()]


def load_notes(input_epub: Path, src_lang: str, tgt_lang: str) -> list[Note]:
    return [n for p in notes_files(input_epub, src_lang, tgt_lang) for n in parse_notes(p)]


def notes_block(notes: list[Note], text: str, tgt_lang: str) -> str:
    """Bloque de notas para el prompt de `text`, o "" si ninguna aplica."""
    hits = [n for n in notes if n.pattern.search(text)]
    if not hits:
        return ""
    header = NOTES_HEADER.get(tgt_lang.split("-")[0].lower(), NOTES_HEADER_DEFAULT)
    return header + "\n" + "".join(f"- {n.text}\n" for n in hits)


def insert_notes(prompt: str, block: str) -> str:
    """Inserta el bloque de notas en el prompt ya renderizado por la plantilla."""
    i = prompt.find(PROMPT_NOTES_ANCHOR)
    if i < 0:
        raise RuntimeError(f"No encontré {PROMPT_NOTES_ANCHOR!r} en el prompt del modelo: "
                           "las notas de traducción no son compatibles con esta plantilla.")
    return prompt[:i] + block + prompt[i:]


# ─── Caché de traducciones (resume) ────────────────────────────────────────────

class TranslationCache:
    """Caché append-only en JSONL: una línea {"k": hash, "t": traducción} por texto.

    Es el único estado persistente de la traducción (docstring, punto 5):
    - Append + flush por entrada: una interrupción pierde como mucho las
      secuencias que se estaban generando, nunca lo ya traducido.
    - Si un texto se vuelve a traducir (p. ej. para corregir una salida mala),
      la línea nueva se agrega al final y, al leer, la última gana.
    - Un texto que el modelo no logró traducir se guarda con "f": 1 y el texto
      original: el libro se puede armar igual (ese párrafo queda sin traducir),
      pero translate_epub.py lo vuelve a intentar en la próxima ejecución.
    - read_only=True permite que build_epub.py la lea mientras translate_epub.py
      sigue escribiendo: solo se agregan líneas, nunca se reescribe el archivo.
    """

    def __init__(self, path: Path, src_lang: str, tgt_lang: str, read_only: bool = False,
                 notes: list[Note] | None = None):
        self.path = path
        self.prefix = f"{MODEL_NAME}|{src_lang}>{tgt_lang}\n"
        self.tgt_lang = tgt_lang
        self.notes = notes or []
        self.data: dict[str, str] = {}
        self.failed_keys: set[str] = set()  # claves guardadas con el original
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self.data[entry["k"]] = entry["t"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue  # línea truncada por una interrupción
                    # La última línea de cada clave gana, también para la marca
                    if entry.get("f"):
                        self.failed_keys.add(entry["k"])
                    else:
                        self.failed_keys.discard(entry["k"])
        self._fh = None
        if read_only:
            return
        self._fh = path.open("a", encoding="utf-8")
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            self._fh.write("\n")  # cerrar línea truncada

    def notes_block(self, text: str) -> str:
        """Notas que aplican a `text` (las que van en su prompt), o ""."""
        return notes_block(self.notes, text, self.tgt_lang)

    def _key(self, text: str) -> str:
        # Incluir modelo e idiomas en la clave evita reutilizar traducciones de
        # otra configuración; el hash mantiene las claves cortas y de largo fijo.
        # Las notas que aplican al texto también son parte de la clave: si se
        # agrega o cambia una nota, solo se retraducen los textos que la usan.
        # Sin notas el bloque es "" y la clave es la de siempre.
        return hashlib.sha1((self.prefix + self.notes_block(text) + text)
                            .encode("utf-8")).hexdigest()

    def get(self, text: str) -> str | None:
        return self.data.get(self._key(text))

    def is_failed(self, text: str) -> bool:
        """¿Lo guardado para `text` es el original porque la traducción falló?"""
        return self._key(text) in self.failed_keys

    def put(self, text: str, translated: str, failed: bool = False):
        key = self._key(text)
        self.data[key] = translated
        entry = {"k": key, "t": translated}
        if failed:
            entry["f"] = 1
            self.failed_keys.add(key)
        else:
            self.failed_keys.discard(key)
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # flush por entrada: el costo es despreciable al lado de la generación
        # (~0.4 s por texto) y garantiza que el progreso quede en disco.
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


# ─── Modelo ────────────────────────────────────────────────────────────────────

class Translator:
    """Traduce listas de textos con batching continuo (BatchGenerator de mlx_lm):
    mantiene BATCH_SIZE secuencias activas y, cuando una termina, la guarda en la
    caché y entra la siguiente (docstring, punto 3).

    El modelo se carga recién al primer uso: si todo ya está en la caché, el
    script termina sin tocar la GPU."""

    def __init__(self, src_lang: str, tgt_lang: str, cache: TranslationCache):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.cache = cache
        self.model = None
        self.tokenizer = None
        self.stop_ids: list[int] = []
        self.generated = 0  # textos traducidos en esta ejecución
        # Comienzo común de todos los prompts y su KV cache (docstring, punto 14)
        self.prefix: list[int] = []
        self.prefix_cache = None

    def _load(self):
        # Import diferido: build_epub.py importa este módulo y no debe cargar
        # mlx (ni inicializar la GPU) para armar un EPUB.
        from mlx_lm import load as mlx_load
        print(f"[model] Cargando {MODEL_NAME}...", flush=True)
        t0 = time.time()
        self.model, self.tokenizer = mlx_load(MODEL_NAME)
        # Gemma termina el turno con <end_of_turn>, que no está en eos_token_ids.
        # Sin esto la generación seguía hasta max_tokens produciendo tokens que
        # luego se descartaban: era la principal causa de lentitud (docstring, punto 1).
        stop = set(self.tokenizer.eos_token_ids)
        eot = self.tokenizer.convert_tokens_to_ids(END_OF_TURN)
        if isinstance(eot, int) and eot != self.tokenizer.unk_token_id:
            stop.add(eot)
        self.stop_ids = sorted(stop)
        self._build_prefix_cache()
        print(f"[model] Listo ({time.time() - t0:.1f}s)\n", flush=True)

    def _prompt(self, text: str) -> tuple[list[int], int]:
        """Devuelve (tokens del prompt, max_tokens para esta salida).

        TranslateGemma recibe los idiomas como campos del mensaje (no como una
        instrucción en texto libre); su plantilla de chat arma el prompt. Las
        notas de traducción que apliquen al texto se insertan en él. Un modelo
        general ("chat") recibe instrucciones de traductor como mensaje de
        sistema, con las notas al final, y el razonamiento desactivado."""
        block = self.cache.notes_block(text)
        n_in = len(self.tokenizer.encode(text))
        max_tokens = int(n_in * MAX_TOKENS_FACTOR) + MAX_TOKENS_MARGIN
        if MODEL_KIND == "chat":
            system = CHAT_SYSTEM_PROMPT.format(src=lang_name(self.src_lang),
                                               tgt=lang_name(self.tgt_lang))
            if block:
                system += "\n\n" + block
            # enable_thinking=False: sin esto Gemma 4 "piensa" antes de responder
            # y gasta el límite de tokens (en la prueba, 52 de 52 salidas cortadas)
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                add_generation_prompt=True, enable_thinking=False)
            return prompt, max_tokens
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": self.src_lang,
                "target_lang_code": self.tgt_lang,
                "text": text,
            }],
        }]
        if block:
            # Con notas se renderiza la plantilla como texto, se insertan y se
            # tokeniza (la plantilla ya incluye <bos>). Sin notas, el camino de
            # siempre: el prompt queda idéntico al de antes.
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True)
            prompt = self.tokenizer.encode(insert_notes(rendered, block),
                                           add_special_tokens=False)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # max_tokens proporcional a la entrada (docstring, punto 4): holgado para
        # una traducción normal, pero acota cuánto puede durar un bucle.
        return prompt, max_tokens

    def _build_prefix_cache(self):
        """Procesa una sola vez el comienzo común de los prompts (docstring,
        punto 14). El comienzo se obtiene comparando prompts de textos distintos:
        así sirve para cualquier plantilla sin conocer su formato."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        samples = [self._prompt(t)[0] for t in ("Erster Text.", "zweiter", "»Dritter«")]
        n = 0
        while all(n < len(p) - 1 and p[n] == samples[0][n] for p in samples):
            n += 1
        if n < 8:  # no vale la pena
            return
        cache = make_prompt_cache(self.model)
        self.model(mx.array([samples[0][:n]]), cache=cache)
        mx.eval([c.state for c in cache])
        self.prefix, self.prefix_cache = samples[0][:n], cache

    def translate_many(self, texts: list[str], failed: set[str]):
        """Traduce `texts` y guarda cada resultado en la caché apenas termina.
        Si la salida es mala (bucle, cortada, demasiado larga) se reintenta una vez
        con penalización de repetición y el doble de max_tokens; si vuelve a
        fallar se guarda el texto original marcado como fallo (se reintenta en
        la próxima ejecución). Los textos con error se agregan a `failed`."""
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_logits_processors
        if self.model is None:
            self._load()
        retry_processors = make_logits_processors(
            repetition_penalty=RETRY_REPETITION_PENALTY,
            repetition_context_size=RETRY_REPETITION_CONTEXT)
        # Pila invertida: pop() saca del final en O(1) y respeta el orden original
        # (orden de lectura). Lo que se agrega al final sale primero (los
        # reintentos se atienden de inmediato).
        # Ventanas en orden de lectura; dentro de cada una, de mayor a menor largo:
        # los textos de largo parecido comparten batch (docstring, punto 14).
        window = BATCH_SIZE * WINDOW_BATCHES
        texts = [t for i in range(0, len(texts), window)
                 for t in sorted(texts[i:i + window], key=len, reverse=True)]
        queue: list[tuple[str, int]] = [(t, 0) for t in reversed(texts)]  # (texto, intento)
        # uid de BatchGenerator -> (texto, intento, tokens generados hasta ahora)
        active: dict[int, tuple[str, int, list[int]]] = {}
        total, done, kept_original = len(texts), 0, 0
        start = last_report = time.time()
        batch = BATCH_SIZE  # se reduce si la GPU se queda sin memoria
        gen = None

        def feed():
            """Rellena los lugares libres del batch. Se llama después de cada paso
            de generación: así ningún lugar queda esperando al párrafo más largo
            (lo que hace "continuo" al batching). Los textos entran de a
            ADMIT_GROUP: procesar 4 prompts juntos cuesta casi lo mismo que
            procesar 1, así que conviene esperar a tener 4 lugares libres."""
            if active and batch - len(active) < min(ADMIT_GROUP, batch, len(queue)):
                return
            while queue and len(active) < batch:
                text, attempt = queue.pop()
                try:
                    prompt, max_tokens = self._prompt(text)
                except Exception as e:
                    print(f"    ⚠ Error preparando prompt: {e}", flush=True)
                    failed.add(text)
                    continue
                # El reintento tiene el doble de margen: si la primera salida se
                # cortó por max_tokens siendo una traducción legítimamente larga,
                # repetir con el mismo límite la cortaría otra vez.
                max_tokens *= 1 + attempt
                # Penalización solo en el reintento: aplicada siempre, también
                # castigaría repeticiones legítimas en traducciones correctas.
                processors = [retry_processors] if attempt else None
                # Si el prompt empieza con el prefijo común, parte de una copia de
                # su KV cache y solo se procesa el resto. Con notas de traducción
                # el comienzo es otro y se procesa entero.
                caches = None
                n = len(self.prefix)
                if self.prefix_cache is not None and prompt[:n] == self.prefix:
                    prompt, caches = prompt[n:], [copy.deepcopy(self.prefix_cache)]
                uid = gen.insert([prompt], [max_tokens], caches=caches,
                                 logits_processors=processors)[0]
                active[uid] = (text, attempt, [])

        def step():
            """Avanza un token en TODAS las secuencias activas (los pesos se leen
            una vez por paso y sirven a todo el batch) y guarda las que terminan."""
            nonlocal done, kept_original, last_report
            for r in gen.next_generated():
                text, attempt, tokens = active[r.uid]
                if r.finish_reason != "stop":
                    tokens.append(r.token)
                if r.finish_reason is None:
                    continue
                del active[r.uid]
                result = clean_output(self.tokenizer.decode(tokens))
                problem = ("cortada por max_tokens" if r.finish_reason == "length"
                           else output_problem(text, result))
                if problem and attempt == 0:
                    print(f"    ⟳ Reintento ({problem}): {text[:60]!r}...", flush=True)
                    queue.append((text, 1))
                    continue
                if problem:
                    print(f"    ⚠ Se conserva el original ({problem}): {text[:60]!r}...",
                          flush=True)
                    result = text
                    kept_original += 1
                    failed.add(text)
                # Se guarda apenas termina, sin esperar al resto del batch.
                # Un fallo se guarda marcado: sirve para armar el libro, y
                # la próxima ejecución lo vuelve a intentar.
                self.cache.put(text, result, failed=bool(problem))
                self.generated += 1
                done += 1
                now = time.time()
                if now - last_report >= PROGRESS_EVERY_S or done == total:
                    last_report = now
                    eta = (now - start) / done * (total - done)
                    print(f"      {done}/{total} textos  "
                          f"({done / (now - start) * 60:.0f}/min, ETA ~{eta / 60:.0f} min)",
                          flush=True)

        while queue or active:
            # completion_batch_size: secuencias generando a la vez (lo que acelera).
            # prefill_batch_size: prompts que se procesan juntos al entrar; el
            # prefill ya es eficiente por sí solo (procesa todo el prompt en
            # paralelo), así que agruparlos de a 8 basta y evita picos de memoria
            # con prompts largos.
            gen = BatchGenerator(self.model, stop_tokens=[[t] for t in self.stop_ids],
                                 completion_batch_size=batch,
                                 prefill_batch_size=min(batch, 8))
            try:
                feed()
                while active:
                    step()
                    feed()
            except RuntimeError as e:
                # Sin memoria en la GPU: lo terminado ya está en la caché. Lo que
                # estaba en curso vuelve a la cola (en su orden) y se sigue con
                # la mitad de secuencias en paralelo.
                if not is_oom(e) or batch == 1:
                    raise
                queue.extend((text, attempt) for text, attempt, _ in
                             reversed(list(active.values())))
                active.clear()
                batch = max(1, batch // 2)
                print(f"    ⚠ GPU sin memoria: se sigue con batch {batch}", flush=True)
            finally:
                gen.close()
                gen = None
                mx.clear_cache()  # devuelve al sistema la memoria del batch
        if kept_original:
            print(f"    ⚠ {kept_original} textos quedaron en el idioma original (ver arriba); "
                  "se reintentan al relanzar")


# ─── Utilidades de texto ───────────────────────────────────────────────────────

def split_ws(text: str) -> tuple[str, str, str]:
    """Separa (espacio inicial, núcleo, espacio final)."""
    core = text.strip()
    if not core:
        return text, "", ""
    start = text.index(core)
    return text[:start], core, text[start + len(core):]


def has_letters(text: str) -> bool:
    return any(c.isalpha() for c in text)


def output_problem(src: str, out: str) -> str | None:
    """Motivo por el que una traducción parece mala, o None si parece correcta.

    Heurísticas baratas para los fallos reales observados (docstring, punto 9):
    - la misma palabra 8+ veces seguidas: bucle de la decodificación greedy;
    - salida > 3x el original + 50: el modelo "explica" en vez de traducir
      (pasa con fragmentos cortos como "th" o "WiE ES"). En el libro de prueba
      ninguna traducción normal de más de 40 caracteres superó 1.9x; el margen
      fijo evita falsos positivos en textos muy cortos.
    Se usa al generar y también al leer la caché, para corregir entradas malas
    de ejecuciones anteriores. Sobre las 3237 traducciones del libro de prueba
    marcó solo los 3 casos malos reales.
    """
    if not out:
        return "salida vacía"
    if re.search(r"\b(\w+)(?:\W+\1\b){7,}", out):
        return "bucle de repetición"
    if len(out) > len(src) * MAX_OUTPUT_RATIO + MAX_OUTPUT_MARGIN:
        return "salida demasiado larga"
    return None


def is_skipped(node) -> bool:
    return any(p.name in SKIP_TAGS for p in node.parents if p.name)


# ─── Unidades de traducción ────────────────────────────────────────────────────
#
# Estrategia por bloque (párrafo, título, celda...), de mejor a peor:
#   1. "md":    cursiva/negrita como Markdown (*x*, **x**), que el modelo respeta
#               en general; la salida se vuelve a convertir en las etiquetas
#               originales (md_to_nodes).
#   2. "plain": bloques sin cursiva/negrita: el texto entero. Si el modelo agrega
#               Markdown por su cuenta (p. ej. pone un título en cursiva), también
#               se convierte en etiquetas: nunca quedan asteriscos en el libro.
#   3. "nodes": bloques con links, notas, <br>, imágenes o ids con texto (las
#               anclas vacías, como los marcadores de página, se reinsertan):
#               cada fragmento de texto por separado, para no perder nada.
#
# Por qué el párrafo es la unidad: el modelo traduce mejor con la oración
# completa (traducir fragmento por fragmento separaba sujeto y verbo cuando
# había una cursiva en el medio), y reemplazar el contenido de cada bloque
# conserva tal cual la estructura de párrafos, clases y estilos del original.
#
# Por qué Markdown: probado con el modelo real, las etiquetas tipo <g1>…</g1>
# o <em>…</em> se descartaban casi siempre, mientras que *…* se conservaba en
# la mayoría de los párrafos. El Markdown es solo el formato de ida y vuelta
# con el modelo: md_to_nodes lo convierte siempre en HTML, aunque el modelo
# haya cambiado algún marcador.

def collect_units(soup) -> list:
    """Devuelve unidades a traducir en orden de documento:
    - Tag: un bloque hoja (párrafo, título, celda...): bloque sin sub-bloques
    - NavigableString: texto suelto fuera de un bloque hoja

    Recorrido recursivo único, O(n): cada llamada devuelve si su subárbol
    contiene un bloque, así el padre sabe si es "hoja" sin volver a recorrer
    sus descendientes (hacerlo por cada bloque sería O(n²) en HTML anidado).
    """
    def walk(node, skipped: bool) -> tuple[bool, list]:
        """Devuelve (contiene un bloque, unidades)."""
        has_block, found = False, []
        for child in node.children:
            if isinstance(child, Tag):
                child_block, child_units = walk(child, skipped or child.name in SKIP_TAGS)
                has_block |= child_block or child.name in BLOCK_TAGS
                found.extend(child_units)
            elif type(child) is NavigableString and not skipped and has_letters(child):
                found.append(child)
        if (isinstance(node, Tag) and node.name in BLOCK_TAGS
                and not has_block and not skipped):
            return False, [node] if has_letters(node.get_text()) else []
        return has_block, found

    return walk(soup, False)[1]


def unwrap_target(block: Tag) -> Tag:
    """Si el bloque solo envuelve un elemento inline (<h4><span>17</span></h4>,
    <p><a href>Kapitel</a></p>), traduce el contenido de ese elemento.

    Así un envoltorio que abarca todo el bloque (un link del índice, un span
    de estilo) se conserva intacto y no obliga a usar el modo "nodes"."""
    target = block
    while True:
        sig = [c for c in target.children
               if not (type(c) is NavigableString and not c.strip())]
        if len(sig) == 1 and isinstance(sig[0], Tag) and sig[0].name in INLINE_TAGS:
            target = sig[0]
        else:
            return target


def is_anchor(tag: Tag) -> bool:
    """Elemento sin texto que solo marca una posición: marcador de página
    (<span epub:type="pagebreak" id="page-18"/>), destino de un link (<a id>)...
    No cuenta br/img/wbr, que sí cambian lo que se ve."""
    return (tag.name not in VISIBLE_EMPTY_TAGS and not tag.get_text().strip()
            and tag.find(VISIBLE_EMPTY_TAGS) is None)


def anchors_in(target: Tag) -> list[tuple[Tag, float]]:
    """Anclas de primer nivel del bloque y su posición relativa (0-1) en el texto."""
    total = sum(len(s) for s in target.find_all(string=True)) or 1
    found, pos = [], 0
    for d in target.descendants:
        if isinstance(d, NavigableString):
            pos += len(d)
        elif is_anchor(d) and not any(is_anchor(p) for p in d.parents
                                      if p is not target and target in p.parents):
            found.append((d, pos / total))
    return found


def insert_at(target: Tag, element: Tag, ratio: float):
    """Inserta `element` en la posición relativa `ratio` del texto de `target`,
    al inicio de la palabra más cercana (nunca en medio de una palabra)."""
    strings = [s for s in target.find_all(string=True) if type(s) is NavigableString]
    total = sum(len(s) for s in strings)
    goal, acc = round(ratio * total), 0
    if goal == 0:  # al inicio del bloque, no dentro de su primer hijo
        target.insert(0, element)
        return
    for s in strings:
        text = str(s)
        if acc + len(text) >= goal:
            k = goal - acc
            if 0 < k < len(text) and not text[k - 1].isspace():
                nxt = re.search(r"\s", text[k:])
                k = k + nxt.end() if nxt else len(text)
            parts = [NavigableString(text[:k]), element, NavigableString(text[k:])]
            s.replace_with(*[x for x in parts if not isinstance(x, NavigableString) or x])
            return
        acc += len(text)
    target.append(element)


def block_mode(target: Tag) -> str:
    """Elige la estrategia del bloque: "md", "plain" o "nodes" (ver arriba).

    "nodes" se reserva para lo que no se puede perder (links, notas, anclas
    con texto), porque traduce peor: cada fragmento llega al modelo sin
    contexto. Las anclas vacías (marcadores de página) no cuentan: se sacan
    antes de traducir y se reinsertan después en la posición equivalente. Un
    texto que ya tiene "*" no puede usar Markdown (sería ambiguo)."""
    anchors = {id(a) for a in target.find_all(True) if is_anchor(a)}
    for t in target.find_all(True):
        if id(t) in anchors or any(id(p) in anchors for p in t.parents):
            continue
        if t.name not in INLINE_TAGS or t.name in PRESERVE_TAGS or t.has_attr("id"):
            return "nodes"
    if any(type(c) is not NavigableString for c in target.find_all(string=True)):
        return "nodes"  # comentarios, CDATA...
    if any(t.name in MD_TAGS for t in target.find_all(True)):
        return "plain" if "*" in target.get_text() else "md"
    return "plain"


def md_elements(target: Tag) -> list[Tag]:
    return [t for t in target.find_all(True)
            if t.name in MD_TAGS and t.get_text().strip()]


def encode_md(target: Tag) -> str:
    """Texto del bloque con cursiva/negrita como *x* / **x**.

    Los espacios del borde del elemento quedan fuera de los marcadores
    ("* Hotel*" no es Markdown válido y el modelo lo rompería). Los inline
    que no son de MD_TAGS (span, u, ...) se vuelven transparentes: su texto
    se traduce y restore_inline() intenta recuperar el elemento después."""
    def walk(node) -> str:
        parts = []
        for child in node.children:
            if isinstance(child, Tag):
                inner = walk(child)
                if child.name in MD_TAGS and child.get_text().strip():
                    lead, core, trail = split_ws(inner)
                    mark = MD_TAGS[child.name]
                    parts.append(f"{lead}{mark}{core}{mark}{trail}")
                else:
                    parts.append(inner)  # inline transparente (span, u, ...)
            else:
                parts.append(str(child))
        return "".join(parts)
    return walk(target)


MD_RE = re.compile(r"\*\*|\*")


def parse_md(text: str) -> list | None:
    """Árbol [texto | (marca, hijos)] si los * de `text` están bien cerrados,
    o None si no (un * suelto o cierres cruzados como *a **b* c**)."""
    root: list = []
    stack: list[tuple[str, list]] = [("", root)]
    pos = 0
    for m in MD_RE.finditer(text):
        if text[pos:m.start()]:
            stack[-1][1].append(text[pos:m.start()])
        pos = m.end()
        mark = m.group(0)
        if stack[-1][0] == mark:
            stack[-2][1].append(stack.pop())
        elif any(open_mark == mark for open_mark, _ in stack[1:]):
            return None
        else:
            stack.append((mark, []))
    if text[pos:]:
        stack[-1][1].append(text[pos:])
    return root if len(stack) == 1 else None


def md_to_nodes(text: str, soup, elements: list[Tag]) -> tuple[list, set[int]]:
    """Convierte la salida del modelo (texto con Markdown) en nodos HTML.
    Devuelve (nodos, ids de los elementos de `elements` reutilizados).

    Tolerante a propósito: el modelo a menudo agrega, quita o mueve algún
    marcador (en el libro de prueba, en ~40% de los párrafos con cursiva).
    Exigir que coincidan exactamente con el original hacía descartar todo el
    formato del párrafo, y aceptar el texto tal cual dejaba asteriscos en el
    libro. En cambio:
    - Cada tramo *x* / **x** usa el siguiente elemento original del mismo tipo
      (copiando nombre y atributos, p. ej. <em class="titulo">); si ya no
      quedan, un <em>/<strong> genérico (el modelo puso en cursiva algo que
      el original marcaba de otra forma, como un título entre ›comillas‹).
    - Si los marcadores no están bien cerrados, se eliminan: se pierde ese
      formato, nunca el texto, y no queda ningún asterisco suelto.
    Los elementos originales que no se reutilizaron los puede recuperar
    restore_inline() si su texto aparece literal en la traducción."""
    tree = parse_md(text)
    if tree is None:
        return [NavigableString(MD_RE.sub("", text))], set()
    pools = {mark: deque(e for e in elements if MD_TAGS[e.name] == mark)
             for mark in set(MD_TAGS.values())}
    used: set[int] = set()

    def plain_text(items) -> str:
        return "".join(it if isinstance(it, str) else plain_text(it[1]) for it in items)

    def build(items) -> list:
        out = []
        for it in items:
            if isinstance(it, str):
                out.append(NavigableString(it))
                continue
            mark, children = it
            inner = build(children)
            if not plain_text(children).strip():
                out.extend(inner)  # "**" vacío: no crear una etiqueta vacía
                continue
            if pools[mark]:
                src = pools[mark].popleft()
                used.add(id(src))
                tag = soup.new_tag(src.name, attrs=dict(src.attrs))
            else:
                tag = soup.new_tag("em" if mark == "*" else "strong")
            for n in inner:
                tag.append(n)
            out.append(tag)
        return out

    return build(tree), used


def leaf_inline(target: Tag) -> list[Tag]:
    """Elementos inline sin hijos y con texto: los candidatos a restore_inline()."""
    return [t for t in target.find_all(True) if not t.find(True) and t.get_text().strip()]


def inside(node, name: str, stop: Tag) -> bool:
    """¿`node` está dentro de un elemento `name`, sin pasar de `stop`?"""
    for p in node.parents:
        if p is stop:
            return False
        if p.name == name:
            return True
    return False


def restore_inline(target: Tag, originals: list[tuple[str, dict, str]], soup):
    """Re-aplica formato inline perdido cuando su texto aparece literal en la
    traducción (siglas en versalitas, números, nombres propios en cursiva...).

    Funciona porque lo que suele llevar formato tipográfico (siglas, cifras,
    títulos de obras) casi nunca se traduce: si el texto exacto aparece en la
    salida, es seguro volver a envolverlo. Si no aparece, se pierde solo el
    estilo, nunca el texto. No envuelve texto que ya está dentro de un
    elemento del mismo tipo (evita <em><em>x</em></em>)."""
    restored: set[int] = set()
    for name, attrs, text in originals:
        pat = re.compile(r"(?<!\w)" + re.escape(text.strip()) + r"(?!\w)")
        for node in target.find_all(string=True):
            if type(node) is not NavigableString or id(node.parent) in restored:
                continue
            if inside(node, name, target):
                continue
            m = pat.search(str(node))
            if not m:
                continue
            new = soup.new_tag(name, attrs=attrs)
            new.string = m.group(0)
            parts = [NavigableString(str(node)[:m.start()]), new,
                     NavigableString(str(node)[m.end():])]
            node.replace_with(*[x for x in parts if not isinstance(x, NavigableString) or x])
            restored.add(id(new))
            break


def fill_block(target: Tag, text: str, soup, source_has_stars: bool):
    """Reemplaza el contenido de `target` por la traducción `text`.

    Si el original no tenía asteriscos, los de la salida son Markdown (nuestro
    o agregado por el modelo) y se convierten en etiquetas; si los tenía (p. ej.
    marcas de notas al pie), se dejan como texto."""
    elements = md_elements(target)
    originals = leaf_inline(target)
    anchors = anchors_in(target)
    for anchor, _ in anchors:
        anchor.extract()
    if source_has_stars:
        nodes, used = [NavigableString(text)], set()
    else:
        nodes, used = md_to_nodes(text, soup, elements)
    target.clear()
    for n in nodes:
        target.append(n)
    restore_inline(target, [(t.name, dict(t.attrs), t.get_text())
                            for t in originals if id(t) not in used], soup)
    # Marcadores de página y anclas: en la misma posición relativa del texto.
    # Sin esto, un párrafo con un marcador de página en medio se traducía por
    # fragmentos: oraciones partidas y palabras pegadas (en un libro hecho
    # desde PDF, 23% de los párrafos tenían uno).
    for anchor, ratio in anchors:
        insert_at(target, anchor, ratio)


def keep_edge_punctuation(source: str, result: str) -> str:
    """Repone la puntuación de los bordes de un fragmento si el modelo la quitó.

    Al traducir un fragmento suelto (modo "nodes"), el modelo tiende a omitir
    la puntuación inicial: ", such as the family" → "como la familia". Sin la
    coma ni el espacio, la palabra anterior (p. ej. una cursiva) quedaba pegada
    ("ministerialescomo"). También repone comas, dos puntos o signos de
    apertura al final, que unen el fragmento con el siguiente."""
    lead = re.match(r"[,.;:!?)\]»”’]+\s*", source)
    if lead and not result.startswith(lead.group(0).rstrip()[0]):
        result = lead.group(0) + result.lstrip()
    trail = re.search(r"\s*[,;:(\[«“‘]$", source)
    if trail and not result.endswith(trail.group(0).strip()):
        result = result.rstrip() + trail.group(0)
    return result


def translate_string(node: NavigableString, lookup: Lookup, soup=None) -> bool:
    # Solo el núcleo va al modelo (que recorta espacios); los espacios del borde
    # se reponen, así no se pegan palabras con la etiqueta vecina.
    lead, core, trail = split_ws(str(node))
    result = lookup(core)
    if result is None:
        return False
    text = lead + keep_edge_punctuation(core, result) + trail
    if "*" in str(node) or not MD_RE.search(result):
        node.replace_with(NavigableString(text))
    elif soup is None or any(p.name in MD_TAGS for p in node.parents if p.name):
        # Ya está dentro de una cursiva/negrita: el Markdown del modelo sobra
        node.replace_with(NavigableString(MD_RE.sub("", text)))
    else:
        node.replace_with(*md_to_nodes(text, soup, [])[0])
    return True


def translate_block(block: Tag, lookup: Lookup, soup) -> bool:
    """Aplica la traducción a un bloque. `lookup(texto)` devuelve la traducción
    o None si todavía no existe (en ese caso el bloque queda sin cambios).
    Devuelve True si el bloque quedó completamente traducido."""
    target = unwrap_target(block)
    mode = block_mode(target)
    if mode in ("md", "plain"):
        source = encode_md(target) if mode == "md" else target.get_text()
        lead, core, trail = split_ws(source)
        result = lookup(core)
        if result is None:
            return False
        fill_block(target, lead + result + trail, soup,
                   source_has_stars="*" in target.get_text())
        return True
    ok = True
    for s in [s for s in target.find_all(string=True)
              if type(s) is NavigableString and has_letters(s) and not is_skipped(s)]:
        ok &= translate_string(s, lookup, soup)
    return ok


# ─── Extracción EPUB ───────────────────────────────────────────────────────────

def extract_epub(epub_path: Path, src_dir: Path):
    """Descomprime el EPUB una sola vez. src/ nunca se modifica: es la fuente
    de verdad tanto para planificar la traducción como para armar el EPUB."""
    if (src_dir / "mimetype").exists():
        print(f"[extract] Ya existe en {src_dir}, skip.")
        return
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {epub_path.name} → {src_dir}")
    with zipfile.ZipFile(epub_path) as zf:
        zf.extractall(src_dir)
    count = sum(1 for f in src_dir.rglob("*") if f.is_file())
    print(f"[extract] {count} archivos.")


# ─── Estructura del EPUB ───────────────────────────────────────────────────────

def book_dirs(input_epub: Path) -> tuple[Path, Path]:
    """(work_dir, src_dir) del libro."""
    work_dir = WORK_ROOT / input_epub.stem
    return work_dir, work_dir / "src"


def model_file(work_dir: Path) -> Path:
    return work_dir / "modelo.txt"


def remembered_model(input_epub: Path) -> str | None:
    """Modelo con el que se tradujo este libro la última vez (o None). Lo usan
    los tres scripts por defecto, para que build_epub.py busque en la caché las
    traducciones del mismo modelo que las hizo."""
    f = model_file(book_dirs(input_epub)[0])
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


def resolve_model(input_epub: Path, requested: str | None) -> str:
    """--modelo si se indicó; si no, el recordado para el libro; si no, el default."""
    remembered = remembered_model(input_epub)
    repo = select_model(requested or remembered or DEFAULT_MODEL)
    if requested and remembered and repo != remembered:
        print(f"⚠ Este libro se tradujo antes con {remembered}; con {repo} se traduce "
              "de nuevo (las traducciones anteriores quedan en la caché).")
    return repo


def cache_path(work_dir: Path, src_lang: str, tgt_lang: str) -> Path:
    return work_dir / f"cache_{src_lang}-{tgt_lang}.jsonl"


def find_opf(src_dir: Path) -> Path:
    container = etree.parse(str(src_dir / "META-INF" / "container.xml"))
    full_path = container.find(".//{*}rootfile").get("full-path")
    return src_dir / unquote(full_path)


def resolve_href(base_file: Path, href: str) -> Path:
    """Ruta absoluta de un href relativo a `base_file` (sin #fragmento)."""
    return Path(posixpath.normpath(str(base_file.parent / unquote(href.split("#")[0]))))


def spine_files(src_dir: Path, include_rest: bool = True) -> list[Path]:
    """HTML en orden de lectura (spine); al final, los HTML fuera del spine.

    El orden importa (docstring, punto 7): traducir siguiendo el spine hace que
    los capítulos se completen de principio a fin, que es lo que necesita un
    EPUB parcial. El orden alfabético de archivos no sirve ("chapter10" va
    antes que "chapter2")."""
    opf = find_opf(src_dir)
    tree = etree.parse(str(opf))
    hrefs = {it.get("id"): it.get("href") for it in tree.iterfind(".//{*}manifest/{*}item")}
    ordered: list[Path] = []
    for ref in tree.iterfind(".//{*}spine/{*}itemref"):
        href = hrefs.get(ref.get("idref"))
        if href:
            path = resolve_href(opf, href)
            if path.suffix.lower() in HTML_EXTENSIONS and path.exists() and path not in ordered:
                ordered.append(path)
    if not include_rest:
        return ordered
    rest = sorted(f for f in src_dir.rglob("*")
                  if f.is_file() and f.suffix.lower() in HTML_EXTENSIONS and f not in ordered)
    return ordered + rest


def find_ncx(src_dir: Path) -> Path | None:
    opf = find_opf(src_dir)
    tree = etree.parse(str(opf))
    for it in tree.iterfind(".//{*}manifest/{*}item"):
        if it.get("media-type") == "application/x-dtbncx+xml":
            return resolve_href(opf, it.get("href"))
    return None


def ncx_label_nodes(ncx_tree) -> list:
    """Elementos <text> de títulos del índice NCX (docTitle y navLabel)."""
    return [t for t in ncx_tree.iterfind(".//{*}text")
            if t.text and has_letters(t.text)]


def opf_title_texts(src_dir: Path) -> list[str]:
    """Títulos del libro en el OPF (<dc:title>). Se piden aparte porque no
    están en ningún HTML: en un EPUB3 sin índice NCX nadie más los pediría y
    build_epub.py dejaría el título en el idioma original."""
    tree = etree.parse(str(find_opf(src_dir)))
    return [split_ws(t.text)[1] for t in tree.iterfind(".//{*}metadata/{*}title")
            if t.text and has_letters(t.text)]


# ─── Confirmación ──────────────────────────────────────────────────────────────

def is_interactive() -> bool:
    """¿Se puede preguntar al usuario? Requiere terminal en la entrada y en la
    salida (si la salida va a un log, la pregunta no se vería) y que el proceso
    esté en primer plano: un proceso lanzado con `&` que lee la terminal queda
    suspendido por el sistema en vez de fallar."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        return os.getpgrp() == os.tcgetpgrp(sys.stdin.fileno())
    except OSError:
        return False


def describe_notes(input_epub: Path, src_lang: str, tgt_lang: str, notes: list[Note]) -> str:
    files = notes_files(input_epub, src_lang, tgt_lang)
    if not files:
        return f"ninguna (puedes crear {NOTES_DIR.name}/{src_lang}-{tgt_lang}.md)"
    return f"{len(notes)} de " + ", ".join(
        f.relative_to(SCRIPT_DIR).as_posix() if f.is_relative_to(SCRIPT_DIR) else f.name
        for f in files)


def confirm_plan(input_epub: Path, src_lang: str, tgt_lang: str, work_dir: Path,
                 n_files: int, units: int, units_pending: int, texts_pending: int,
                 notes_desc: str, texts_with_notes: int) -> bool:
    """Muestra qué se va a hacer y pregunta si empezar. Sin terminal, termina
    con un error que indica usar -q (nunca se queda esperando una respuesta)."""
    est_min = texts_pending * REF_SECONDS_PER_TEXT / 60
    print("Plan de traducción")
    print(f"  Libro:      {input_epub.name} ({n_files} archivos HTML)")
    print(f"  Idiomas:    {src_lang} → {tgt_lang}")
    print(f"  Modelo:     {MODEL_NAME} (batch {BATCH_SIZE})")
    print(f"  Trabajo:    {work_dir}")
    print(f"  Notas:      {notes_desc}")
    print(f"  Unidades:   {units} en total, {units - units_pending} ya traducidas, "
          f"{units_pending} pendientes")
    print(f"  A traducir: {texts_pending} textos, ~{max(est_min, 1):.0f} min estimados "
          f"(referencia: {REF_SECONDS_PER_TEXT} s/texto en M5 Pro, batch 32)")
    if texts_with_notes:
        print(f"              {texts_with_notes} de ellos llevan notas de traducción "
              "(incluye los que se rehacen por una nota nueva o cambiada)")
    print("  Se puede interrumpir con Ctrl+C y retomar después sin perder lo traducido.")
    if not is_interactive():
        print("\n❌ No hay una terminal para confirmar (¿en segundo plano o con la salida "
              "a un archivo?). Usa -q para empezar sin preguntar.")
        sys.exit(2)
    try:
        answer = input("\n¿Empezar la traducción? [s/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"s", "si", "sí", "y", "yes"}


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def apply_translations(html_file: Path, lookup: Lookup):
    """Parsea un HTML y aplica las traducciones disponibles.
    Devuelve (soup, texto_original, unidades, unidades_incompletas).

    Lo usan las dos fases: run() para planificar (con un lookup que anota lo que
    falta) y build_epub.py para armar el libro (con cache.get). Compartir el
    código garantiza que las claves pedidas y las buscadas coincidan."""
    text = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(text, "html.parser")
    units = collect_units(soup)
    incomplete = 0
    for u in units:
        ok = (translate_block(u, lookup, soup) if isinstance(u, Tag)
              else translate_string(u, lookup, soup))
        incomplete += not ok
    return soup, text, len(units), incomplete


def run(input_epub: Path, src_lang: str, tgt_lang: str, ask: bool = True) -> bool:
    """Extrae el EPUB y traduce a la caché todo lo que falte.
    Con `ask`, antes de cargar el modelo muestra el plan y pide confirmación.
    Devuelve False si el usuario decide no empezar."""
    work_dir, src_dir = book_dirs(input_epub)
    created_work_dir = not work_dir.exists()
    extract_epub(input_epub, src_dir)

    notes = load_notes(input_epub, src_lang, tgt_lang)
    notes_desc = describe_notes(input_epub, src_lang, tgt_lang, notes)
    if not ask:
        print(f"[notes] {notes_desc}", flush=True)
    cache = TranslationCache(cache_path(work_dir, src_lang, tgt_lang), src_lang, tgt_lang,
                             notes=notes)
    tr = Translator(src_lang, tgt_lang, cache)
    html_files = spine_files(src_dir)
    ncx = find_ncx(src_dir)
    ncx_texts = ([split_ws(t.text)[1] for t in ncx_label_nodes(etree.parse(str(ncx)))]
                 if ncx and ncx.exists() else [])
    title_texts = opf_title_texts(src_dir)

    # Rondas: cada una recorre el libro con un lookup que anota lo que falta en la
    # caché (en orden de lectura, sin duplicados) y lo traduce. Normalmente basta
    # una; otra ronda solo hace falta si alguna traducción se rehízo y cambió lo
    # que se pide. Máximo 3 para no ciclar.
    start = time.time()
    failed: set[str] = set()
    try:
        for round_no in range(1, 4):
            missing: dict[str, None] = {}  # dict = conjunto que conserva el orden

            def lookup(text: str) -> str | None:
                result = cache.get(text)
                if result is not None and output_problem(text, result):
                    # Traducción mala de una ejecución anterior (antes de existir
                    # la validación): se vuelve a pedir y la nueva la reemplaza.
                    result = None
                # `failed` evita pedir en cada ronda un texto que ya falló en esta.
                # Los que fallaron en una ejecución anterior (guardados con el
                # original) sí se piden de nuevo, una vez por ejecución.
                if (result is None or cache.is_failed(text)) and text not in failed:
                    missing[text] = None
                return result

            # Título del libro y títulos del índice primero (cortos, útiles en un parcial)
            for t in title_texts + ncx_texts:
                lookup(t)
            total = incomplete = 0
            for f in html_files:
                _, _, n, inc = apply_translations(f, lookup)
                total += n
                incomplete += inc
            if not missing:
                break
            # La confirmación va después de planificar (para mostrar cuánto falta)
            # y antes de cargar el modelo, que es lo costoso.
            if ask and round_no == 1 and not confirm_plan(
                    input_epub, src_lang, tgt_lang, work_dir, len(html_files),
                    total, incomplete, len(missing), notes_desc,
                    sum(1 for t in missing if cache.notes_block(t))):
                cache.close()
                if created_work_dir:
                    shutil.rmtree(work_dir)  # no dejar rastros de un libro que no se tradujo
                print("Cancelado. No se tradujo nada.")
                return False
            print(f"[translate] ronda {round_no}: {total} unidades, "
                  f"{len(missing)} textos pendientes.", flush=True)
            model_file(work_dir).write_text(MODEL_NAME + "\n", encoding="utf-8")
            tr.translate_many(list(missing), failed)
    except KeyboardInterrupt:
        # Ctrl+C es seguro: todo lo terminado ya está en disco (flush por entrada).
        cache.close()
        print("\n⏸  Interrumpido. El progreso está guardado; relanza el mismo comando para seguir.")
        sys.exit(130)
    cache.close()

    elapsed = time.time() - start
    print(f"\n✅ Traducción terminada en {elapsed / 60:.1f} min "
          f"({tr.generated} textos traducidos ahora, {len(failed)} con error)")
    if failed:
        print("   ⚠ Esos textos quedan en el idioma original en el EPUB; "
              "relanza para reintentarlos.")
    return True


def add_model_argument(parser: argparse.ArgumentParser):
    choices = "; ".join(f"{k}: {m.summary}" for k, m in MODELS.items())
    parser.add_argument("--modelo", metavar="MODELO",
                        help=f"alias o repositorio MLX. {choices}. Por defecto, el último usado "
                             f"con este libro, o {DEFAULT_MODEL}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae y traduce un EPUB con un modelo local (MLX). "
                    "El EPUB se arma con build_epub.py.")
    parser.add_argument("input", type=Path, help="EPUB de entrada")
    parser.add_argument("--src", required=True,
                        help="idioma origen, código ISO (ej. de, en, fr, pt-BR); obligatorio")
    parser.add_argument("--tgt", default=DEFAULT_TARGET_LANG, help="idioma destino (ej. es)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"párrafos en paralelo (default {BATCH_SIZE})")
    add_model_argument(parser)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="empezar sin mostrar el plan ni pedir confirmación "
                             "(necesario en segundo plano o con la salida a un log)")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch debe ser >= 1")
    globals()["BATCH_SIZE"] = args.batch  # translate_many lee el valor global

    input_epub = args.input.resolve()
    if not input_epub.exists():
        print(f"❌ No existe: {input_epub}")
        sys.exit(1)

    resolve_model(input_epub, args.modelo)
    print(f"Input:  {input_epub}")
    print(f"Idioma: {args.src} → {args.tgt}")
    print(f"Model:  {MODEL_NAME} (MLX, batch {BATCH_SIZE})")
    print(f"Work:   {book_dirs(input_epub)[0]}\n")

    if not run(input_epub, args.src, args.tgt, ask=not args.quiet):
        return
    tgt = "" if args.tgt == DEFAULT_TARGET_LANG else f" --tgt {args.tgt}"
    print(f"   Para armar el EPUB: .venv/bin/python build_epub.py "
          f"{shlex.quote(str(args.input))} --src {args.src}{tgt}")


if __name__ == "__main__":
    main()
