#!/usr/bin/env python3
"""
Arma un EPUB con las traducciones disponibles en la caché de translate_epub.py.

- Si el libro está completo: EPUB traducido completo.
- Si no: EPUB parcial con los capítulos traducidos en orden de lectura, cortando
  en el primer capítulo incompleto (spine, manifest e índice se ajustan).

Puede correr mientras translate_epub.py sigue traduciendo (lee la caché sin
modificarla).

Diseño
──────
- Sin modelo: solo lee la caché y aplica las traducciones con el mismo código
  que usa translate_epub.py para planificar (apply_translations), así que las
  claves buscadas coinciden siempre con las traducidas. Arma un libro en ~1 s.
- Siempre parte de work/<libro>/src/ (el original intacto) y escribe en build/,
  que se regenera en cada ejecución: armar el EPUB no tiene estado propio y se
  puede repetir cuantas veces se quiera.
- Corta en el PRIMER capítulo incompleto del spine (no saltea huecos): un
  libro con capítulos faltantes en el medio sería confuso de leer.
- Los archivos sin texto (portada SVG) se copian sin re-serializar, porque
  html.parser pasaría a minúsculas atributos SVG como viewBox y rompería la imagen.
  Si la portada SVG tiene texto (un <title>), se traduce y restore_svg_case
  devuelve las mayúsculas a las etiquetas y atributos SVG.
- fit_images agrega a cada CSS del libro un alto máximo para las imágenes, para
  que en una ventana baja no se corten y sigan en la página siguiente.

Uso:
    .venv/bin/python build_epub.py <entrada.epub> [salida.epub] --src <idioma> [--tgt es]

Ejemplo:
    .venv/bin/python build_epub.py input.epub --src de
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

from translate_epub import (
    DEFAULT_TARGET_LANG, TranslationCache, apply_translations,
    add_model_argument, book_dirs, cache_path, find_ncx, find_opf, load_notes,
    ncx_label_nodes, resolve_href, resolve_model, spine_files, split_ws,
)

DOCTYPE_XHTML = ('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" '
                 '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n')
PARTIAL_SUFFIX = " (parcial)"
# Agregado a cada CSS del libro: muchos EPUB no limitan el alto de las
# imágenes y, en una ventana baja, una imagen se corta y sigue en la página
# siguiente. En Apple Books 100vh es la ventana entera: se descuentan ~160px
# de su título y número de página, y 2em de márgenes del propio libro.
FIT_IMAGES_CSS = ("\n/* epub translator: las imágenes caben en la página */\n"
                  "img { max-width: 100%; max-height: calc(100vh - 160px - 2em); object-fit: contain; }\n")

# html.parser pasa a minúsculas nombres de etiquetas y atributos, pero en SVG
# (XML) importan las mayúsculas: sin viewBox/preserveAspectRatio la portada se
# deforma. Nombres SVG con mayúsculas, para restaurarlos al re-serializar.
SVG_NAMES = {n.lower(): n for n in (
    "viewBox preserveAspectRatio gradientUnits gradientTransform patternUnits "
    "patternContentUnits patternTransform clipPathUnits maskUnits maskContentUnits "
    "markerUnits markerWidth markerHeight refX refY textLength lengthAdjust "
    "startOffset spreadMethod stdDeviation baseFrequency numOctaves "
    "primitiveUnits filterUnits pathLength attributeName repeatCount "
    "linearGradient radialGradient clipPath foreignObject textPath "
    "feGaussianBlur feColorMatrix feOffset feBlend feFlood feComposite feMerge "
    "feMergeNode feImage feTurbulence animateTransform"
).split()}


# ─── Serialización XML válida ──────────────────────────────────────────────────

def get_doctype(text: str) -> str:
    """DOCTYPE del archivo original (XHTML 1.1, EPUB3 `<!DOCTYPE html>`...),
    para no imponer uno distinto al del libro."""
    m = re.search(r"<!DOCTYPE[^>]*>", text, re.IGNORECASE)
    return (re.sub(r"\s+", " ", m.group(0)) + "\n") if m else DOCTYPE_XHTML


def fit_images(build_dir: Path):
    """Agrega FIT_IMAGES_CSS a todas las hojas de estilo del libro."""
    for css in build_dir.rglob("*.css"):
        with css.open("a", encoding="utf-8") as f:
            f.write(FIT_IMAGES_CSS)


def restore_svg_case(tree):
    """Devuelve a SVG las mayúsculas que html.parser le quitó (ver SVG_NAMES)."""
    for el in tree.iter("{http://www.w3.org/2000/svg}*"):
        ns, _, local = el.tag.rpartition("}")
        el.tag = ns + "}" + SVG_NAMES.get(local, local)
        for name in [a for a in el.attrib if a in SVG_NAMES]:
            el.set(SVG_NAMES[name], el.attrib.pop(name))


def fix_xhtml(text: str, doctype: str) -> bytes:
    """Re-serializa como XML válido (DOCTYPE, namespaces, self-closing).

    BeautifulSoup con html.parser es tolerante para leer, pero escribe HTML,
    no XHTML; los lectores EPUB (Apple Books en particular) exigen XML bien
    formado. lxml lo re-serializa como XML estricto."""
    # html.parser deja el DOCTYPE roto como texto "html PUBLIC ..."
    text = re.sub(r'^\s*(<!DOCTYPE[^>]*>|html\s+(?:PUBLIC|SYSTEM)\s[^\n]*)\n?', '',
                  re.sub(r'^\s*<\?xml[^>]*\?>\s*', '', text), flags=re.IGNORECASE)
    text = '<?xml version="1.0" encoding="utf-8"?>\n' + text
    try:
        tree = etree.fromstring(text.encode("utf-8"))
        restore_svg_case(tree)
        result = etree.tostring(tree, xml_declaration=True, encoding="utf-8")
    except etree.XMLSyntaxError as e:
        print(f"    ⚠ XML inválido, se guarda sin re-serializar: {e}", flush=True)
        result = text.encode("utf-8")
    end_decl = result.find(b"?>") + 2
    return result[:end_decl] + b"\n" + doctype.encode() + result[end_decl:]


# ─── OPF e índice ──────────────────────────────────────────────────────────────

def translate_text(text: str, cache: TranslationCache) -> str:
    """Traducción de un texto suelto (título, entrada del índice) o el original
    si no está en la caché."""
    lead, core, trail = split_ws(text)
    result = cache.get(core)
    return lead + result + trail if result else text


def update_opf(opf: Path, dropped: set[Path], cache: TranslationCache, tgt_lang: str,
               partial: bool) -> set[Path]:
    """Ajusta idioma, título, spine, manifest y guide. Devuelve los archivos
    del manifest marcados como nav (EPUB3)."""
    tree = etree.parse(str(opf))
    nav_files: set[Path] = set()
    dropped_ids = set()
    for item in list(tree.iterfind(".//{*}manifest/{*}item")):
        path = resolve_href(opf, item.get("href", ""))
        if path in dropped:
            dropped_ids.add(item.get("id"))
            item.getparent().remove(item)
        elif "nav" in (item.get("properties") or "").split():
            nav_files.add(path)
    for ref in list(tree.iterfind(".//{*}spine/{*}itemref")):
        if ref.get("idref") in dropped_ids:
            ref.getparent().remove(ref)
    for ref in list(tree.iterfind(".//{*}guide/{*}reference")):
        if resolve_href(opf, ref.get("href", "")) in dropped:
            ref.getparent().remove(ref)
    for lang in tree.iterfind(".//{*}metadata/{*}language"):
        lang.text = tgt_lang
    for title in tree.iterfind(".//{*}metadata/{*}title"):
        if title.text:
            title.text = translate_text(title.text, cache) + (PARTIAL_SUFFIX if partial else "")
    tree.write(str(opf), xml_declaration=True, encoding="utf-8")
    return nav_files


def update_ncx(ncx: Path, dropped: set[Path], cache: TranslationCache, partial: bool):
    """Traduce títulos del índice y quita entradas a capítulos descartados
    (sus sub-entradas que sigan existiendo suben un nivel)."""
    tree = etree.parse(str(ncx))
    for t in ncx_label_nodes(tree):
        t.text = translate_text(t.text, cache)
    if partial:
        for t in tree.iterfind(".//{*}docTitle/{*}text"):
            t.text = (t.text or "") + PARTIAL_SUFFIX
    # De adentro hacia afuera, para que los hijos ya estén resueltos. Un capítulo
    # descartado puede contener sub-entradas que sí se conservan (p. ej. la
    # portadilla agrupa "[Buch]" y "[Autor]"): suben un nivel en vez de perderse.
    for nav_point in reversed(list(tree.iterfind(".//{*}navPoint"))):
        content = nav_point.find("{*}content")
        if content is None or resolve_href(ncx, content.get("src", "")) not in dropped:
            continue
        parent = nav_point.getparent()
        pos = parent.index(nav_point)
        for child in reversed(nav_point.findall("{*}navPoint")):
            parent.insert(pos + 1, child)
        parent.remove(nav_point)
    for i, nav_point in enumerate(tree.iterfind(".//{*}navPoint"), 1):
        nav_point.set("playOrder", str(i))
    tree.write(str(ncx), xml_declaration=True, encoding="utf-8")


def prune_nav(soup, nav_file: Path, dropped: set[Path]):
    """Quita del nav EPUB3 los <li> que apuntan a capítulos descartados."""
    for a in soup.find_all("a", href=True):
        if resolve_href(nav_file, a["href"]) in dropped:
            li = a.find_parent("li")
            (li or a).decompose()


# ─── Ensamblado EPUB ───────────────────────────────────────────────────────────

def assemble_epub(build_dir: Path, output_path: Path):
    """Comprime build/ como EPUB.

    La especificación exige que `mimetype` sea el primer archivo del zip y sin
    comprimir: así los lectores identifican el formato leyendo los primeros
    bytes. Se escribe a un .tmp y se renombra al final, para que un corte a
    mitad de camino nunca deje un EPUB roto en el lugar del bueno."""
    tmp = output_path.with_suffix(".epub.tmp")
    all_files = sorted(f for f in build_dir.rglob("*") if f.is_file() and f.name != "mimetype")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), b"application/epub+zip",
                    compress_type=zipfile.ZIP_STORED)
        for f in all_files:
            zf.write(f, str(f.relative_to(build_dir)), compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(output_path)
    size = output_path.stat().st_size / 1024
    print(f"[epub] {output_path.name} ({size:.0f} KB, {len(all_files) + 1} archivos)")


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def build(input_epub: Path, output_epub: Path | None, src_lang: str, tgt_lang: str):
    work_dir, src_dir = book_dirs(input_epub)
    cpath = cache_path(work_dir, src_lang, tgt_lang)
    if not (src_dir / "mimetype").exists() or not cpath.exists():
        print(f"❌ No hay traducciones para {input_epub.name}. Corre primero translate_epub.py.")
        sys.exit(1)
    # Las mismas notas que usó translate_epub.py: forman parte de la clave de caché
    cache = TranslationCache(cpath, src_lang, tgt_lang, read_only=True,
                             notes=load_notes(input_epub, src_lang, tgt_lang))

    # Aplicar traducciones en orden de lectura y buscar el primer capítulo incompleto.
    # Un archivo sin texto (n == 0) cuenta como completo: no corta en la portada.
    # Textos que el modelo no logró traducir: están en la caché con el original
    # (el libro se arma igual), pero se avisa para que no pasen inadvertidos.
    untranslated: set[str] = set()

    def lookup(text: str) -> str | None:
        if cache.is_failed(text):
            untranslated.add(text)
        return cache.get(text)

    spine = spine_files(src_dir, include_rest=False)
    rendered = {}
    cut = len(spine)
    units_total = units_done = 0
    for i, f in enumerate(spine):
        soup, text, n, incomplete = apply_translations(f, lookup)
        rendered[f] = (soup, text, n)
        units_total += n
        units_done += n - incomplete
        if incomplete and cut == len(spine):
            cut = i
    for f in spine_files(src_dir):
        if f not in rendered:  # HTML fuera del spine (nav, etc.)
            soup, text, n, _ = apply_translations(f, lookup)
            rendered[f] = (soup, text, n)

    partial = cut < len(spine)
    kept = spine[:cut]
    if not any(rendered[f][2] for f in kept):
        print("❌ Todavía no hay ningún capítulo completamente traducido.")
        sys.exit(1)
    dropped = set(spine[cut:])

    if output_epub is None:
        suffix = f"{tgt_lang.upper()} parcial" if partial else tgt_lang.upper()
        output_epub = input_epub.parent / f"{input_epub.stem} ({suffix}).epub"

    build_dir = work_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(src_dir, build_dir)
    fit_images(build_dir)

    def in_build(path: Path) -> Path:
        return build_dir / path.relative_to(src_dir)

    dropped_build = {in_build(p) for p in dropped}
    nav_files = update_opf(in_build(find_opf(src_dir)), dropped_build, cache, tgt_lang, partial)
    ncx = find_ncx(src_dir)
    if ncx and ncx.exists():
        update_ncx(in_build(ncx), dropped_build, cache, partial)

    for f, (soup, text, n) in rendered.items():
        target = in_build(f)
        if f in dropped:
            target.unlink()
            continue
        if target in nav_files:
            prune_nav(soup, target, dropped_build)
        elif not n:
            continue  # sin texto (p. ej. portada SVG): se deja el original intacto
        target.write_bytes(fix_xhtml(str(soup), get_doctype(text)))

    print(f"[build] {'Parcial' if partial else 'Completo'}: {len(kept)}/{len(spine)} "
          f"archivos del spine"
          + (f", corta antes de {spine[cut].relative_to(src_dir)}" if partial else "")
          + f". Unidades traducidas: {units_done}/{units_total} "
            f"({units_done / max(units_total, 1):.0%} del libro).")
    if untranslated:
        print(f"[build] ⚠ {len(untranslated)} textos quedaron en el idioma original porque "
              "el modelo no logró traducirlos; translate_epub.py los reintenta al relanzarlo.")
    assemble_epub(build_dir, output_epub)
    print(f"   Resultado: {output_epub}")


def main():
    parser = argparse.ArgumentParser(
        description="Arma un EPUB (completo o parcial) con las traducciones disponibles.")
    parser.add_argument("input", type=Path, help="EPUB original (el mismo de translate_epub.py)")
    parser.add_argument("output", type=Path, nargs="?", help="EPUB de salida")
    parser.add_argument("--src", required=True,
                        help="idioma origen, código ISO (ej. de, en, fr, pt-BR); obligatorio")
    parser.add_argument("--tgt", default=DEFAULT_TARGET_LANG, help="idioma destino (ej. es)")
    add_model_argument(parser)
    args = parser.parse_args()

    input_epub = args.input.resolve()
    if not input_epub.exists():
        print(f"❌ No existe: {input_epub}")
        sys.exit(1)
    print(f"Modelo: {resolve_model(input_epub, args.modelo)}")
    build(input_epub, args.output.resolve() if args.output else None, args.src, args.tgt)


if __name__ == "__main__":
    main()
