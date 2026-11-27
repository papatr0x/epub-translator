#!/usr/bin/env python3
"""
Detecta palabras que el modelo probablemente traduce mal (acepciones dudosas,
coloquialismos, jerga) y propone notas de traducción para revisar antes de
traducir el libro.

Resultado: <libro>.notas.propuesta.md junto al EPUB. Ese archivo NO lo usa el
traductor: revisa cada propuesta, corrígela o bórrala, y copia las que sirvan
a <libro>.notas.md (solo este libro) o a notas/<src>-<tgt>.md (todos los libros).

Cómo funciona
─────────────
1. Candidatas, sin modelo: palabras que aparecen varias veces en el libro pero
   son raras en el idioma general (frecuencias de `wordfreq`). Las formas de una
   misma palabra (knipste, geknipst...) se agrupan quitando terminaciones.
2. Prueba con el mismo modelo, por cada candidata y en hasta N apariciones:
   - la oración suelta, con la palabra marcada entre corchetes ([knipste]); el
     modelo suele devolver marcada su traducción, y eso dice cómo la entiende
     con poco contexto (en las pruebas conservó la marca en ~35-40% de los
     casos; las que la pierden no cuentan);
   - el párrafo completo, tal como lo traducirá el libro.
   Si la palabra con la que tradujo la oración suelta no aparece en la traducción
   del párrafo, el modelo cambió de acepción con el contexto largo: es justo
   donde se equivoca (así fallaba "knipsen": bien en oraciones sueltas, "golpeó"
   o "jugaba" dentro de párrafos largos). También se marcan las palabras que
   reciben 3 o más traducciones distintas.
3. Propuesta: para cada palabra marcada, una nota con la traducción más común en
   oraciones sueltas, más estadísticas y ejemplos en un comentario.

Es una ayuda para encontrar candidatas, no un juez: algunas marcas son
variantes aceptables (sinónimos) y la acepción propuesta puede estar mal si el
modelo no conoce la palabra (p. ej. jerga de época). Por eso se revisa.

Las traducciones de párrafos quedan en la caché del libro y se reutilizan al
traducir; las de oraciones sueltas van a una caché aparte. Retomable.

Uso:
    .venv/bin/python detect_terms.py <entrada.epub> --src <idioma> [--tgt es]
        [--candidatas 80] [--muestras 6] [--batch 16]
"""

import argparse
import collections
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Callable

import translate_epub as T

# ─── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CANDIDATES = 80
DEFAULT_SAMPLES = 6
MIN_OCCURRENCES = 3      # apariciones mínimas en el libro
MAX_ZIPF = 4.0           # más frecuente que esto en el idioma general = palabra común
MIN_STEM = 4             # largo mínimo de la raíz de una familia de palabras
SENTENCE_LEN = (20, 300)  # largo de las oraciones de prueba (caracteres)
# Marca de la palabra en la oración de prueba. Probado con el modelo: corchetes
# 39%, *x* 31%, **x** 28%, «x» 17%, <<x>> 15% de marcas conservadas.
MARK_OPEN, MARK_CLOSE = "[", "]"
MARK_RE = re.compile(r"\[([^\[\]]+)\]")
# Criterios para marcar una palabra como dudosa
MIN_DIVERGENT = 2        # apariciones donde el párrafo no usa la traducción suelta
MIN_SENSES = 3           # traducciones distintas en oraciones sueltas
COPIED_SHARE = 0.6       # si el modelo deja la palabra sin traducir en ≥60%: nombre propio

WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
# Fin de oración: puntuación final, comillas opcionales, espacio y mayúscula
SENTENCE_RE = re.compile(r"(?<=[.!?…])[»«“”\"']?\s+(?=[»«„“\"'(]?[^\W\d_])")
# Terminaciones que se quitan para agrupar formas de una palabra. Pensadas para
# alemán; en otros idiomas agrupan peor (cada forma queda como su propia familia).
SUFFIXES = sorted(["en", "ern", "es", "em", "er", "e", "n", "s", "st", "t", "te", "ten",
                   "tet", "test", "tes", "ter"], key=len, reverse=True)
# Palabras vacías del idioma destino que no sirven para comparar traducciones
STOPWORDS = {
    "es": set("el la los las un una unos unas de del al a en con por para que se lo le les "
              "su sus mi tu muy mas y o no es era fue hizo hacer hace".split()),
    "en": set("the a an of to in on at by for with and or not is was did do make made".split()),
}

PROPOSAL_HEADER = """# Propuesta de notas de traducción: {book}

Generada por detect_terms.py ({date}). **El traductor no usa este archivo.**
Revisa cada nota: corrige la acepción, bórrala si no hace falta, y copia las que
sirvan a `{book_notes}` (solo este libro) o a `notas/{src}-{tgt}.md` (todos los libros).
Los comentarios `<!-- -->` se pueden dejar: el traductor los ignora.

"""


# ─── Candidatas ────────────────────────────────────────────────────────────────

class Candidate:
    def __init__(self, stem: str, forms: collections.Counter, zipf: float, cap_share: float):
        self.stem = stem
        self.forms = forms            # forma en minúsculas -> apariciones
        self.count = sum(forms.values())
        self.zipf = zipf
        self.cap_share = cap_share    # fracción de apariciones con mayúscula inicial
        self.score = self.count * (MAX_ZIPF - zipf)

    def pattern(self) -> str:
        """Patrón para la nota: la raíz como subcadena (también encuentra
        compuestos). Las raíces cortas se anclan al inicio de palabra para no
        coincidir dentro de palabras no relacionadas."""
        escaped = re.escape(self.stem)
        return escaped if len(self.stem) >= 6 else r"\b" + escaped


def strip_suffix(word: str) -> str:
    for s in SUFFIXES:
        if word.endswith(s) and len(word) - len(s) >= MIN_STEM:
            return word[:-len(s)]
    return word


def find_candidates(texts: list[str], src_lang: str, limit: int,
                    notes: list[T.Note]) -> list[Candidate]:
    """Familias de palabras frecuentes en el libro y raras en el idioma general.

    Se reparte el cupo entre palabras que suelen ir en minúscula (verbos,
    adjetivos: donde más se confunden las acepciones) y en mayúscula (nombres,
    sustantivos alemanes, términos de época); los nombres propios los descarta
    la prueba con el modelo, porque los deja sin traducir."""
    from wordfreq import zipf_frequency

    counts: collections.Counter = collections.Counter()
    caps: collections.Counter = collections.Counter()
    for t in texts:
        for w in WORD_RE.findall(t):
            counts[w.lower()] += 1
            caps[w.lower()] += w[0].isupper()

    families: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for w, c in counts.items():
        families[strip_suffix(w)][w] += c
    # Participios alemanes: ge+raíz se suma a la raíz (geknipst -> knips)
    for stem in list(families):
        if stem.startswith("ge") and stem[2:] in families and len(stem[2:]) >= MIN_STEM:
            families[stem[2:]].update(families.pop(stem))
    # Raíces que difieren en la última letra (knipst -> knip, knipsen -> knips)
    for stem in sorted(families, key=len, reverse=True):
        shorter = stem[:-1]
        if len(shorter) >= MIN_STEM and shorter in families and stem in families:
            families[shorter].update(families.pop(stem))

    rows = []
    for stem, forms in families.items():
        n = sum(forms.values())
        if n < MIN_OCCURRENCES or len(stem) < MIN_STEM:
            continue
        if any(note.pattern.search(w) for note in notes for w in forms):
            continue  # ya tiene una nota
        zipf = max(zipf_frequency(w, src_lang) for w in forms)
        if zipf >= MAX_ZIPF:
            continue
        rows.append(Candidate(stem, forms, zipf, sum(caps[w] for w in forms) / n))
    rows.sort(key=lambda c: -c.score)
    lower = [c for c in rows if c.cap_share < 0.5]
    upper = [c for c in rows if c.cap_share >= 0.5]
    half = limit // 2
    picked = lower[:half] + upper[:limit - len(lower[:half])]
    if len(picked) < limit:
        picked += lower[half:half + limit - len(picked)]
    return sorted(picked, key=lambda c: -c.score)


# ─── Prueba con el modelo ──────────────────────────────────────────────────────

class Occurrence:
    def __init__(self, unit: str, sentence: str, marked: str, word: str):
        self.unit = unit          # texto del párrafo tal como se traduce (clave de caché)
        self.sentence = sentence  # oración original (sin marca)
        self.marked = marked      # oración con la palabra entre corchetes
        self.word = word          # forma de la palabra en el texto


def sample_occurrences(cand: Candidate, units: list[str], samples: int) -> list[Occurrence]:
    """Hasta `samples` apariciones en párrafos distintos, en orden de lectura."""
    found = []
    for unit in units:
        plain = unit.replace("*", "")
        for sentence in SENTENCE_RE.split(plain):
            sentence = sentence.strip()
            if not (SENTENCE_LEN[0] <= len(sentence) <= SENTENCE_LEN[1]):
                continue
            if MARK_OPEN in sentence or MARK_CLOSE in sentence:
                continue
            m = next((m for m in WORD_RE.finditer(sentence) if m.group(0).lower() in cand.forms),
                     None)
            if m:
                marked = (sentence[:m.start()] + MARK_OPEN + m.group(0) + MARK_CLOSE
                          + sentence[m.end():])
                found.append(Occurrence(unit, sentence, marked, m.group(0)))
                break
        if len(found) >= samples:
            break
    return found


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def sense_keys(text: str, tgt_lang: str) -> set[str]:
    """Raíces (4 letras) de las palabras con contenido de una traducción. Dos
    traducciones con alguna raíz en común se consideran la misma acepción
    ("fotografió" / "tomó una foto" comparten "foto")."""
    stop = STOPWORDS.get(tgt_lang.split("-")[0], set())
    return {fold(w)[:4] for w in WORD_RE.findall(text) if len(w) >= 3 and fold(w) not in stop}


class Finding:
    def __init__(self, cand: Candidate, occs: list[Occurrence], renderings: list[str | None],
                 paragraphs: list[str], tgt_lang: str):
        self.cand = cand
        self.items = list(zip(occs, renderings, paragraphs))
        usable = [(o, r, p) for o, r, p in self.items if r]
        self.copied = sum(1 for o, r, p in usable if fold(r).startswith(fold(cand.stem)[:5]))
        self.divergent = [(o, r, p) for o, r, p in usable
                          if not sense_keys(r, tgt_lang) & sense_keys(p, tgt_lang)
                          and sense_keys(r, tgt_lang)]
        # Agrupar traducciones por acepción
        groups: list[tuple[set[str], list[str]]] = []
        for _, r, _ in usable:
            keys = sense_keys(r, tgt_lang) or {fold(r)}
            for g in groups:
                if g[0] & keys:
                    g[0].update(keys)
                    g[1].append(r)
                    break
            else:
                groups.append((set(keys), [r]))
        self.senses = sorted((g[1] for g in groups), key=len, reverse=True)
        self.usable = len(usable)

    @property
    def is_name(self) -> bool:
        return self.usable > 0 and self.copied >= self.usable * COPIED_SHARE

    @property
    def flagged(self) -> bool:
        if self.is_name:
            return False
        return len(self.divergent) >= MIN_DIVERGENT or len(self.senses) >= MIN_SENSES

    def reason(self) -> str:
        parts = []
        if len(self.divergent) >= MIN_DIVERGENT:
            parts.append(f"en {len(self.divergent)} de {self.usable} párrafos el modelo cambió de "
                         "acepción respecto de la oración suelta")
        if len(self.senses) >= MIN_SENSES:
            parts.append(f"{len(self.senses)} traducciones distintas")
        return "; ".join(parts)


Translate = Callable[[T.TranslationCache, list[str]], None]


def probe(cands: list[Candidate], units: list[str], samples: int, book_cache: T.TranslationCache,
          probe_cache: T.TranslationCache, translate: Translate, tgt_lang: str) -> list[Finding]:
    """Traduce oraciones marcadas (caché de prueba) y párrafos (caché del libro,
    reutilizada después al traducir) y arma un Finding por candidata."""
    occs = {c.stem: sample_occurrences(c, units, samples) for c in cands}
    sentences = list(dict.fromkeys(o.marked for os_ in occs.values() for o in os_))
    paragraphs = list(dict.fromkeys(o.unit for os_ in occs.values() for o in os_))
    todo_s = [s for s in sentences if probe_cache.get(s) is None]
    todo_p = [p for p in paragraphs if book_cache.get(p) is None]
    print(f"[detect] {len(cands)} candidatas: {len(sentences)} oraciones "
          f"({len(todo_s)} por traducir) y {len(paragraphs)} párrafos "
          f"({len(todo_p)} por traducir).", flush=True)
    if todo_s:
        translate(probe_cache, todo_s)
    if todo_p:
        translate(book_cache, todo_p)
    findings = []
    for c in cands:
        os_ = occs[c.stem]
        rend = []
        for o in os_:
            m = MARK_RE.search(probe_cache.get(o.marked) or "")
            rend.append(m.group(1).strip() if m else None)
        findings.append(Finding(c, os_, rend, [book_cache.get(o.unit) or "" for o in os_],
                                tgt_lang))
    return findings


# ─── Propuesta ─────────────────────────────────────────────────────────────────

def sentence_translation(probe_cache: T.TranslationCache, occ: Occurrence) -> str:
    return MARK_RE.sub(r"\1", probe_cache.get(occ.marked) or "")


def proposal_entry(f: Finding, probe_cache: T.TranslationCache) -> str:
    c = f.cand
    forms = ", ".join(w for w, _ in c.forms.most_common(5))
    main = c.forms.most_common(1)[0][0]
    seen = ", ".join(f"«{g[0]}» ({len(g)})" for g in f.senses)
    best = f.senses[0][0] if f.senses else "…"
    others = [g[0] for g in f.senses[1:3]]
    lines = [f"## {c.pattern()}",
             f"<!-- REVISAR · {c.count} apariciones ({forms}) · {f.reason()}",
             f"     Traducciones en oraciones sueltas: {seen}"]
    for occ, rendering, paragraph in f.items[:3]:
        lines.append(f"     · «{occ.sentence[:160]}»")
        lines.append(f"       → {sentence_translation(probe_cache, occ)[:160]}")
    if f.divergent:
        occ, rendering, paragraph = f.divergent[0]
        lines.append(f"     En el párrafo, «{rendering}» no aparece: …{paragraph[:200]}…")
    lines.append("-->")
    text = (f"«{main}» y sus formas ({forms}) significan aquí «{best}»: tradúcelo así de forma "
            "coherente en todo el libro.")
    if others:
        text += " No como " + " ni ".join(f"«{o}»" for o in others) + "."
    lines.append(text)
    return "\n".join(lines) + "\n"


def proposal_path(input_epub: Path) -> Path:
    return input_epub.with_name(f"{input_epub.stem}.notas.propuesta.md")


def write_proposal(input_epub: Path, src_lang: str, tgt_lang: str, findings: list[Finding],
                   probe_cache: T.TranslationCache) -> Path:
    path = proposal_path(input_epub)
    flagged = [f for f in findings if f.flagged]
    header = PROPOSAL_HEADER.format(book=input_epub.name, date=time.strftime("%Y-%m-%d"),
                                    book_notes=f"{input_epub.stem}.notas.md",
                                    src=src_lang, tgt=tgt_lang)
    body = "\n".join(proposal_entry(f, probe_cache) for f in flagged)
    if not flagged:
        body = "No se encontraron palabras dudosas entre las candidatas revisadas.\n"
    path.write_text(header + body, encoding="utf-8")
    return path


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def book_units(src_dir: Path) -> list[str]:
    """Textos que se envían al modelo (claves de caché), en orden de lectura."""
    units: list[str] = []

    def lookup(text: str) -> None:
        units.append(text)
        return None

    for f in T.spine_files(src_dir):
        T.apply_translations(f, lookup)
    return list(dict.fromkeys(units))


def run(input_epub: Path, src_lang: str, tgt_lang: str, limit: int, samples: int) -> Path:
    work_dir, src_dir = T.book_dirs(input_epub)
    T.extract_epub(input_epub, src_dir)
    notes = T.load_notes(input_epub, src_lang, tgt_lang)
    units = book_units(src_dir)
    cands = find_candidates([u.replace("*", "") for u in units], src_lang, limit, notes)
    print(f"[detect] {len(units)} textos en el libro; {len(cands)} candidatas "
          f"(sin contar {len(notes)} con nota).", flush=True)

    book_cache = T.TranslationCache(T.cache_path(work_dir, src_lang, tgt_lang), src_lang,
                                    tgt_lang, notes=notes)
    # Sin notas: se quiere ver cómo entiende el modelo cada palabra por sí solo
    probe_cache = T.TranslationCache(work_dir / f"detect_{src_lang}-{tgt_lang}.jsonl",
                                     src_lang, tgt_lang)
    translator = T.Translator(src_lang, tgt_lang, book_cache)

    def translate(cache: T.TranslationCache, texts: list[str]):
        translator.cache = cache
        translator.translate_many(texts, set())

    try:
        findings = probe(cands, units, samples, book_cache, probe_cache, translate, tgt_lang)
    except KeyboardInterrupt:
        print("\n⏸  Interrumpido. Lo traducido quedó guardado; relanza para seguir.")
        sys.exit(130)
    finally:
        book_cache.close()
        probe_cache.close()

    path = write_proposal(input_epub, src_lang, tgt_lang, findings, probe_cache)
    flagged = [f for f in findings if f.flagged]
    names = sum(1 for f in findings if f.is_name)
    print(f"\n✅ {len(flagged)} palabras dudosas de {len(findings)} candidatas "
          f"({names} parecen nombres propios y se descartaron).")
    for f in flagged:
        print(f"   · {f.cand.forms.most_common(1)[0][0]:20s} {f.reason()}")
    print(f"\n   Propuesta: {path}")
    print("   Revísala y copia las notas que sirvan a "
          f"{input_epub.stem}.notas.md o a notas/{src_lang}-{tgt_lang}.md antes de traducir.")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Detecta palabras dudosas y propone notas de traducción para revisar.")
    parser.add_argument("input", type=Path, help="EPUB de entrada")
    parser.add_argument("--src", required=True,
                        help="idioma origen, código ISO (ej. de, en, fr, pt-BR); obligatorio")
    parser.add_argument("--tgt", default=T.DEFAULT_TARGET_LANG, help="idioma destino (ej. es)")
    parser.add_argument("--candidatas", type=int, default=DEFAULT_CANDIDATES,
                        help=f"palabras a revisar (default {DEFAULT_CANDIDATES})")
    parser.add_argument("--muestras", type=int, default=DEFAULT_SAMPLES,
                        help=f"apariciones a probar por palabra (default {DEFAULT_SAMPLES})")
    parser.add_argument("--batch", type=int, default=T.BATCH_SIZE,
                        help=f"textos en paralelo (default {T.BATCH_SIZE})")
    T.add_model_argument(parser)
    args = parser.parse_args()
    T.BATCH_SIZE = args.batch

    input_epub = args.input.resolve()
    if not input_epub.exists():
        print(f"❌ No existe: {input_epub}")
        sys.exit(1)
    print(f"Modelo: {T.resolve_model(input_epub, args.modelo)}")
    run(input_epub, args.src, args.tgt, args.candidatas, args.muestras)


if __name__ == "__main__":
    main()
