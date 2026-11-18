#!/usr/bin/env python3
"""
Traduce un EPUB usando TranslateGemma 4B (MLX, Apple Silicon, gratis).

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
   Medido en un M5 Pro: 1.30 s/párrafo secuencial → 0.36 s/párrafo con 16.

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

7. Orden de lectura + respaldos inmediatos  (spine_files, on_md/on_done)
   Se traduce en el orden del spine, con los títulos del índice primero. Si el
   modelo rompe la cursiva de un párrafo, su respaldo en texto plano se encola
   al momento, no al final del libro. Así cada capítulo queda completo apenas
   se traduce, y build_epub.py puede armar un EPUB parcial en cualquier momento.

8. Unidad = párrafo, formato como Markdown  (sección "Unidades de traducción")
   Traducir el párrafo entero da contexto al modelo y conserva la estructura.
   Las etiquetas inline se pasan como Markdown (*x*, **x**) porque el modelo
   lo vio mucho en su entrenamiento y lo respeta; las etiquetas tipo XML
   (<g1>...</g1>) las descartaba casi siempre.

9. Validación y reintento  (output_problem, translate_many)
   La decodificación greedy a veces cae en bucles ("die die die...") o, ante
   fragmentos sin contexto, el modelo "explica" en vez de traducir. Se detecta
   por heurística y se reintenta una vez con penalización de repetición. La
   penalización solo se usa en el reintento porque en general también castiga
   repeticiones legítimas (nombres propios, estribillos). Si el reintento
   también falla, se conserva el original.

10. Recorrido O(n) del HTML  (collect_units)
    Un solo recorrido recursivo decide qué bloques son "hoja" en vez de
    reescanear los descendientes de cada bloque (O(n²) en HTML anidado).

11. mlx_lm se importa solo dentro de Translator
    build_epub.py reutiliza este módulo sin cargar el modelo ni la GPU.

Uso:
    .venv/bin/python translate_epub.py <entrada.epub> [--src de] [--tgt es] [--batch 16]

Ejemplo:
    .venv/bin/python translate_epub.py input.epub
"""

import argparse
import hashlib
import json
import posixpath
import re
import shlex
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

# ─── Config ────────────────────────────────────────────────────────────────────

# 4 bits: la mitad de bytes a leer por token que en 8 bits (ver docstring, punto 2).
# Cambiarlo invalida la caché (el nombre del modelo es parte de la clave).
MODEL_NAME = "mlx-community/translategemma-4b-it-4bit"
DEFAULT_SOURCE_LANG = "de"
DEFAULT_TARGET_LANG = "es"

# max_tokens de salida = tokens de entrada * factor + margen (docstring, punto 4)
MAX_TOKENS_FACTOR = 2.5
MAX_TOKENS_MARGIN = 64
# Secuencias generadas en paralelo (docstring, punto 3). 16 ≈ 3.5x más rápido que
# secuencial; 32 rinde ~10% más, pero el KV cache (memoria que guarda el contexto
# de cada secuencia activa) crece con el número de secuencias y se duplica.
BATCH_SIZE = 16
PROGRESS_EVERY_S = 30
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
# on_md(texto_md, marcadores_esperados, texto_plano_de_respaldo)
OnMd = Callable[[str, list, str], None]

# Inline que se traduce como Markdown (y se reconstruye)
MD_TAGS = {"em": "*", "i": "*", "cite": "*", "dfn": "*", "var": "*",
           "b": "**", "strong": "**"}
# Inline que obliga a traducir fragmento por fragmento para no perderlo
# (links, notas al pie, saltos de línea e imágenes no se pueden expresar en
# Markdown de forma que el modelo los respete)
PRESERVE_TAGS = {"a", "sup", "sub", "br", "img", "wbr", "code", "kbd", "samp"}


# ─── Caché de traducciones (resume) ────────────────────────────────────────────

class TranslationCache:
    """Caché append-only en JSONL: una línea {"k": hash, "t": traducción} por texto.

    Es el único estado persistente de la traducción (docstring, punto 5):
    - Append + flush por entrada: una interrupción pierde como mucho las
      secuencias que se estaban generando, nunca lo ya traducido.
    - Si un texto se vuelve a traducir (p. ej. para corregir una salida mala),
      la línea nueva se agrega al final y, al leer, la última gana.
    - read_only=True permite que build_epub.py la lea mientras translate_epub.py
      sigue escribiendo: solo se agregan líneas, nunca se reescribe el archivo.
    """

    def __init__(self, path: Path, src_lang: str, tgt_lang: str, read_only: bool = False):
        self.path = path
        self.prefix = f"{MODEL_NAME}|{src_lang}>{tgt_lang}\n"
        self.data: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self.data[entry["k"]] = entry["t"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # línea truncada por una interrupción
        self._fh = None
        if read_only:
            return
        self._fh = path.open("a", encoding="utf-8")
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            self._fh.write("\n")  # cerrar línea truncada

    def _key(self, text: str) -> str:
        # Incluir modelo e idiomas en la clave evita reutilizar traducciones de
        # otra configuración; el hash mantiene las claves cortas y de largo fijo.
        return hashlib.sha1((self.prefix + text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> str | None:
        return self.data.get(self._key(text))

    def put(self, text: str, translated: str):
        key = self._key(text)
        self.data[key] = translated
        self._fh.write(json.dumps({"k": key, "t": translated}, ensure_ascii=False) + "\n")
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
        print(f"[model] Listo ({time.time() - t0:.1f}s)\n", flush=True)

    def _prompt(self, text: str) -> tuple[list[int], int]:
        """Devuelve (tokens del prompt, max_tokens para esta salida).

        TranslateGemma recibe los idiomas como campos del mensaje (no como una
        instrucción en texto libre); su plantilla de chat arma el prompt."""
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": self.src_lang,
                "target_lang_code": self.tgt_lang,
                "text": text,
            }],
        }]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # max_tokens proporcional a la entrada (docstring, punto 4): holgado para
        # una traducción normal, pero acota cuánto puede durar un bucle.
        n_in = len(self.tokenizer.encode(text))
        return prompt, int(n_in * MAX_TOKENS_FACTOR) + MAX_TOKENS_MARGIN

    def translate_many(self, texts: list[str], failed: set[str],
                       on_done: Callable[[str, str], list[str]] | None = None):
        """Traduce `texts` y guarda cada resultado en la caché apenas termina.
        Si la salida es mala (bucle, cortada, demasiado larga) se reintenta una vez
        con penalización de repetición; si vuelve a fallar se conserva el texto
        original. `on_done(texto, resultado)` puede devolver textos extra, que se
        traducen a continuación. Los textos con error se agregan a `failed`."""
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_logits_processors
        if self.model is None:
            self._load()
        # completion_batch_size: secuencias generando a la vez (lo que acelera).
        # prefill_batch_size: prompts que se procesan juntos al entrar; el prefill
        # ya es eficiente por sí solo (procesa todo el prompt en paralelo), así
        # que agruparlos de a 8 basta y evita picos de memoria con prompts largos.
        gen = BatchGenerator(self.model, stop_tokens=[[t] for t in self.stop_ids],
                             completion_batch_size=BATCH_SIZE,
                             prefill_batch_size=min(BATCH_SIZE, 8))
        retry_processors = make_logits_processors(
            repetition_penalty=RETRY_REPETITION_PENALTY,
            repetition_context_size=RETRY_REPETITION_CONTEXT)
        # Pila invertida: pop() saca del final en O(1) y respeta el orden original
        # (orden de lectura). Lo que se agrega al final sale primero (reintentos y
        # respaldos se atienden de inmediato).
        queue: list[tuple[str, int]] = [(t, 0) for t in reversed(texts)]  # (texto, intento)
        # uid de BatchGenerator -> (texto, intento, tokens generados hasta ahora)
        active: dict[int, tuple[str, int, list[int]]] = {}
        total, done, kept_original = len(texts), 0, 0
        start = last_report = time.time()

        def feed():
            """Rellena los lugares libres del batch. Se llama después de cada paso
            de generación: así nunca queda un lugar ocioso esperando al párrafo
            más largo (lo que hace "continuo" al batching)."""
            while queue and len(active) < BATCH_SIZE:
                text, attempt = queue.pop()
                try:
                    prompt, max_tokens = self._prompt(text)
                except Exception as e:
                    print(f"    ⚠ Error preparando prompt: {e}", flush=True)
                    failed.add(text)
                    continue
                # Penalización solo en el reintento: aplicada siempre, también
                # castigaría repeticiones legítimas en traducciones correctas.
                processors = [retry_processors] if attempt else None
                uid = gen.insert([prompt], [max_tokens], logits_processors=processors)[0]
                active[uid] = (text, attempt, [])

        try:
            feed()
            while active:
                # Cada llamada avanza un token en TODAS las secuencias activas:
                # los pesos se leen una vez por paso y sirven a todo el batch.
                for r in gen.next_generated():
                    text, attempt, tokens = active[r.uid]
                    if r.finish_reason != "stop":
                        tokens.append(r.token)
                    if r.finish_reason is None:
                        continue
                    del active[r.uid]
                    result = self.tokenizer.decode(tokens).split(END_OF_TURN)[0].strip()
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
                    # Se guarda apenas termina, sin esperar al resto del batch.
                    self.cache.put(text, result)
                    self.generated += 1
                    # on_done permite encolar al instante el respaldo de un párrafo
                    # cuyo Markdown llegó roto (docstring, punto 7).
                    extra = on_done(text, result) if on_done else []
                    queue.extend((t, 0) for t in reversed(extra))  # van primero
                    total += len(extra)
                    done += 1
                    now = time.time()
                    if now - last_report >= PROGRESS_EVERY_S or done == total:
                        last_report = now
                        eta = (now - start) / done * (total - done)
                        print(f"      {done}/{total} textos  "
                              f"({done / (now - start) * 60:.0f}/min, ETA ~{eta / 60:.0f} min)",
                              flush=True)
                feed()
        finally:
            gen.close()
        if kept_original:
            print(f"    ⚠ {kept_original} textos quedaron en el idioma original (ver arriba)")


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
#   1. "md":    cursiva/negrita como Markdown (*x*, **x**), que el modelo respeta;
#               se valida la salida y se reconstruyen los elementos originales.
#   2. "plain": si el modelo rompe los marcadores, el bloque entero como texto
#               plano (se pierde el formato inline de ese bloque, no la calidad).
#   3. "nodes": bloques con links, notas, <br>, imágenes o anclas (id):
#               cada fragmento de texto por separado, para no perder nada.
#
# Por qué el párrafo es la unidad: el modelo traduce mejor con la oración
# completa (traducir fragmento por fragmento separaba sujeto y verbo cuando
# había una cursiva en el medio), y reemplazar el contenido de cada bloque
# conserva tal cual la estructura de párrafos, clases y estilos del original.
#
# Por qué Markdown: probado con el modelo real, las etiquetas tipo <g1>…</g1>
# o <em>…</em> se descartaban casi siempre, mientras que *…* se conservaba en
# la mayoría de los párrafos. Cuando no se conserva, md_marks_valid/decode_md
# lo detectan y se usa el respaldo "plain".

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


def block_mode(target: Tag) -> str:
    """Elige la estrategia del bloque: "md", "plain" o "nodes" (ver arriba).

    "nodes" se reserva para lo que no se puede perder (links, notas, anclas),
    porque traduce peor: cada fragmento llega al modelo sin contexto. Un
    texto que ya tiene "*" no puede usar Markdown (sería ambiguo)."""
    for t in target.find_all(True):
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


def md_marks_valid(text: str, marks: list[str]) -> bool:
    """¿El Markdown de `text` abre/cierra exactamente los marcadores `marks`
    (en orden de apertura)?

    Es la misma validación que decode_md, pero solo necesita los marcadores
    (no el árbol HTML): translate_many la usa al terminar cada texto para
    decidir al instante si hace falta el respaldo en texto plano."""
    stack: list[str] = []
    opened = 0
    for m in re.finditer(r"\*\*|\*", text):
        mark = m.group(0)
        if stack and stack[-1] == mark:
            stack.pop()
        elif opened < len(marks) and marks[opened] == mark:
            stack.append(mark)
            opened += 1
        else:
            return False
    return not stack and opened == len(marks)


def decode_md(text: str, elements: list[Tag], soup) -> list | None:
    """Reconstruye nodos desde Markdown. None si no coincide con `elements`.

    El n-ésimo marcador que se abre corresponde al n-ésimo elemento original,
    del que se copian nombre y atributos (clases, estilos). Si el modelo quitó,
    agregó o cambió algún marcador, no hay correspondencia segura y se
    devuelve None en vez de adivinar."""
    root: list = []
    stack: list[tuple[str, list]] = [("", root)]
    opened = 0
    opened_index: list[int] = []
    pos = 0
    for m in re.finditer(r"\*\*|\*", text):
        if text[pos:m.start()]:
            stack[-1][1].append(NavigableString(text[pos:m.start()]))
        pos = m.end()
        mark = m.group(0)
        if stack[-1][0] == mark:
            _, children = stack.pop()
            src = elements[opened_index.pop()]
            new = soup.new_tag(src.name, attrs=dict(src.attrs))
            for c in children:
                new.append(c)
            stack[-1][1].append(new)
        else:
            if opened >= len(elements) or MD_TAGS[elements[opened].name] != mark:
                return None
            opened_index.append(opened)
            opened += 1
            stack.append((mark, []))
    if text[pos:]:
        stack[-1][1].append(NavigableString(text[pos:]))
    if len(stack) != 1 or opened != len(elements):
        return None
    return root


def leaf_inline(target: Tag, exclude_md: bool) -> list[tuple[str, dict, str]]:
    """Elementos inline sin hijos (nombre, attrs, texto) que el modo md/plain
    no reconstruye: se intentan restaurar después con restore_inline()."""
    return [(t.name, dict(t.attrs), t.get_text()) for t in target.find_all(True)
            if not t.find(True) and t.get_text().strip()
            and not (exclude_md and t.name in MD_TAGS)]


def restore_inline(target: Tag, originals: list[tuple[str, dict, str]], soup):
    """Re-aplica formato inline perdido cuando su texto aparece literal en la
    traducción (siglas en versalitas, números, nombres propios en cursiva...).

    Funciona porque lo que suele llevar formato tipográfico (siglas, cifras,
    títulos de obras) casi nunca se traduce: si el texto exacto aparece en la
    salida, es seguro volver a envolverlo. Si no aparece, se pierde solo el
    estilo, nunca el texto."""
    restored: set[int] = set()
    for name, attrs, text in originals:
        pat = re.compile(r"(?<!\w)" + re.escape(text.strip()) + r"(?!\w)")
        for node in target.find_all(string=True):
            if type(node) is not NavigableString or id(node.parent) in restored:
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


def translate_string(node: NavigableString, lookup: Lookup) -> bool:
    # Solo el núcleo va al modelo (que recorta espacios); los espacios del borde
    # se reponen, así no se pegan palabras con la etiqueta vecina.
    lead, core, trail = split_ws(str(node))
    result = lookup(core)
    if result is None:
        return False
    node.replace_with(NavigableString(lead + result + trail))
    return True


def translate_plain(target: Tag, lookup: Lookup, soup) -> bool:
    lead, core, trail = split_ws(target.get_text())
    result = lookup(core)
    if result is None:
        return False
    originals = leaf_inline(target, exclude_md=False)
    target.clear()
    target.append(NavigableString(lead + result + trail))
    restore_inline(target, originals, soup)
    return True


def translate_block(block: Tag, lookup: Lookup, soup, on_md: OnMd | None = None) -> bool:
    """Aplica la traducción a un bloque. `lookup(texto)` devuelve la traducción
    o None si todavía no existe (en ese caso el bloque queda sin cambios).
    Devuelve True si el bloque quedó completamente traducido.

    `on_md` avisa qué marcadores espera cada texto md y cuál sería su
    respaldo, para que run() pueda pedir el respaldo apenas llegue un
    resultado con el Markdown roto (docstring, punto 7)."""
    target = unwrap_target(block)
    mode = block_mode(target)
    if mode == "md":
        lead, core, trail = split_ws(encode_md(target))
        elements = md_elements(target)
        if on_md:
            on_md(core, [MD_TAGS[e.name] for e in elements], split_ws(target.get_text())[1])
        result = lookup(core)
        if result is None:
            return False
        nodes = decode_md(lead + result + trail, elements, soup)
        if nodes is not None:
            originals = leaf_inline(target, exclude_md=True)
            target.clear()
            for n in nodes:
                target.append(n)
            restore_inline(target, originals, soup)
            return True
        mode = "plain"  # el modelo rompió los marcadores
    if mode == "plain":
        return translate_plain(target, lookup, soup)
    ok = True
    for s in [s for s in target.find_all(string=True)
              if type(s) is NavigableString and has_letters(s) and not is_skipped(s)]:
        ok &= translate_string(s, lookup)
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


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def apply_translations(html_file: Path, lookup: Lookup, on_md: OnMd | None = None):
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
        ok = (translate_block(u, lookup, soup, on_md) if isinstance(u, Tag)
              else translate_string(u, lookup))
        incomplete += not ok
    return soup, text, len(units), incomplete


def run(input_epub: Path, src_lang: str, tgt_lang: str):
    """Extrae el EPUB y traduce a la caché todo lo que falte."""
    work_dir, src_dir = book_dirs(input_epub)
    extract_epub(input_epub, src_dir)

    cache = TranslationCache(cache_path(work_dir, src_lang, tgt_lang), src_lang, tgt_lang)
    tr = Translator(src_lang, tgt_lang, cache)
    html_files = spine_files(src_dir)
    ncx = find_ncx(src_dir)
    ncx_texts = ([split_ws(t.text)[1] for t in ncx_label_nodes(etree.parse(str(ncx)))]
                 if ncx and ncx.exists() else [])

    # Por cada bloque md pendiente: marcadores esperados y su respaldo en texto plano,
    # para pedir el respaldo apenas el modelo rompa el formato (sin esperar otra ronda).
    md_info: dict[str, tuple[list[str], str]] = {}

    def on_md(core: str, marks: list[str], plain: str):
        md_info[core] = (marks, plain)

    def on_done(text: str, result: str) -> list[str]:
        info = md_info.get(text)
        if info and not md_marks_valid(result, info[0]) and cache.get(info[1]) is None:
            return [info[1]]
        return []

    # Rondas: cada una recorre el libro con un lookup que anota lo que falta en la
    # caché (en orden de lectura, sin duplicados) y lo traduce. Normalmente basta
    # una: los respaldos se encolan durante la misma ronda gracias a on_done. Las
    # siguientes cubren casos como un md roto cacheado en una ejecución anterior,
    # cuyo respaldo recién aparece al volver a aplicar. Máximo 3 para no ciclar.
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
                if result is None and text not in failed:
                    missing[text] = None
                return result

            for t in ncx_texts:  # títulos del índice primero (cortos, útiles en un parcial)
                lookup(t)
            total = 0
            for f in html_files:
                total += apply_translations(f, lookup, on_md)[2]
            if not missing:
                break
            print(f"[translate] ronda {round_no}: {total} unidades, "
                  f"{len(missing)} textos pendientes.", flush=True)
            tr.translate_many(list(missing), failed, on_done)
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
        print("   ⚠ Hay textos sin traducir; relanza para reintentarlos.")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae y traduce un EPUB con TranslateGemma (MLX). "
                    "El EPUB se arma con build_epub.py.")
    parser.add_argument("input", type=Path, help="EPUB de entrada")
    parser.add_argument("--src", default=DEFAULT_SOURCE_LANG, help="idioma origen (ej. de)")
    parser.add_argument("--tgt", default=DEFAULT_TARGET_LANG, help="idioma destino (ej. es)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"párrafos en paralelo (default {BATCH_SIZE})")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch debe ser >= 1")
    globals()["BATCH_SIZE"] = args.batch  # translate_many lee el valor global

    input_epub = args.input.resolve()
    if not input_epub.exists():
        print(f"❌ No existe: {input_epub}")
        sys.exit(1)

    print(f"Input:  {input_epub}")
    print(f"Idioma: {args.src} → {args.tgt}")
    print(f"Model:  {MODEL_NAME} (MLX, batch {BATCH_SIZE})")
    print(f"Work:   {book_dirs(input_epub)[0]}\n")

    run(input_epub, args.src, args.tgt)
    print(f"   Para armar el EPUB: .venv/bin/python build_epub.py {shlex.quote(str(args.input))}")


if __name__ == "__main__":
    main()
