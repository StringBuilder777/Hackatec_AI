"""Skill engine: aprende patrones del usuario y genera código Python optimizado.

Flujo:
1. observe(query, response): cuenta repeticiones por hash de keywords
2. Cuando un patrón llega a CODEGEN_THRESHOLD repeticiones → llama a Gemini CODE
3. find_skill(query): busca skill con score (overlap * confianza) > 0.6
4. run_skill(skill, query, datos): carga el .py con importlib y ejecuta run()
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import importlib.util
import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .gemini import gemini, Modelo


CODEGEN_THRESHOLD = 3
SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "generated"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = SKILLS_DIR / "_index.json"

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "y", "o", "a", "que", "qué", "cual", "cuál", "como", "cómo",
    "para", "por", "con", "sin", "es", "son", "fue", "ser", "este", "esta",
    "ese", "esa", "mi", "mis", "tu", "tus", "su", "sus", "se", "lo", "le",
    "muestra", "muéstrame", "dame", "quiero", "ver", "mostrar",
}


CODEGEN_PROMPT = """Eres un experto en Python y análisis de datos.
El usuario repite este tipo de query con keywords: {keys}
Ejemplos reales:
{examples}

Genera un módulo Python con exactamente esta función:

def run(query: str, datos: list | dict) -> dict:
    # debe retornar: {{ "metricas": [...], "interpretacion": "...", "graficas": [...] }}
    # usar solo pandas, numpy, stdlib
    # ser más rápido que llamar a Gemini

El esquema de retorno es:
{{
  "metricas": [{{"label": str, "valor": str}}],
  "interpretacion": str,
  "graficas": [{{
    "titulo": str,
    "tipo": "bar|line|pie|doughnut|horizontal_bar",
    "datos": {{"labels": [str], "datasets": [{{"label": str, "data": [float]}}]}}
  }}]
}}

Responde SOLO con el bloque ```python ... ``` sin texto extra."""


@dataclass
class Skill:
    pkey: str               # patrón hash
    keys: list[str]         # keywords del patrón
    file: str               # nombre del .py
    confianza: float = 0.6
    veces_usado: int = 0
    veces_exitoso: int = 0
    creado_en: float = field(default_factory=time.time)


@dataclass
class Pattern:
    pkey: str
    keys: list[str]
    frecuencia: int = 0
    examples: list[str] = field(default_factory=list)


def _extract_keywords(query: str) -> list[str]:
    """Tokeniza, baja a minúsculas, filtra stopwords y palabras cortas."""
    txt = query.lower()
    tokens = re.findall(r"[a-záéíóúñü0-9_]+", txt)
    keys = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    # ordenadas y únicas para hash estable
    return sorted(set(keys))


def _hash_keys(keys: list[str]) -> str:
    blob = "|".join(sorted(keys))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


def _strip_python_block(text: str) -> str | None:
    """Extrae el bloque ```python ... ``` de la respuesta de Gemini."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Si no hay backticks, asumir que todo es código
    if "def run" in text:
        return text.strip()
    return None


class SkillStore:
    """Catálogo en memoria de skills aprendidos + persistencia en _index.json."""

    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}
        self.patterns: dict[str, Pattern] = {}
        self._lock = asyncio.Lock()
        self._codegen_in_flight: set[str] = set()
        self._load_index()

    # ─── persistencia índice ──────────────────────────────────────────────

    def _load_index(self) -> None:
        if not INDEX_PATH.exists():
            return
        try:
            data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            for s in data.get("skills", []):
                self.skills[s["pkey"]] = Skill(**s)
            for p in data.get("patterns", []):
                self.patterns[p["pkey"]] = Pattern(**p)
            print(f"[SkillStore] cargados {len(self.skills)} skills, {len(self.patterns)} patrones")
        except Exception as e:
            print(f"[SkillStore] error cargando índice: {e}")

    def _save_index(self) -> None:
        try:
            data = {
                "skills": [asdict(s) for s in self.skills.values()],
                "patterns": [asdict(p) for p in self.patterns.values()],
            }
            INDEX_PATH.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[SkillStore] error guardando índice: {e}")

    # ─── observe / find / run ─────────────────────────────────────────────

    async def observe(
        self,
        query: str,
        response: dict | None = None,
        accepted: bool = True,
    ) -> None:
        """Acumula frecuencia del patrón. Dispara codegen si llega al umbral."""
        keys = _extract_keywords(query)
        if not keys:
            return
        pkey = _hash_keys(keys)

        async with self._lock:
            pat = self.patterns.get(pkey)
            if not pat:
                pat = Pattern(pkey=pkey, keys=keys, frecuencia=0, examples=[])
                self.patterns[pkey] = pat
            pat.frecuencia += 1
            if query not in pat.examples:
                pat.examples.append(query)
                pat.examples = pat.examples[-5:]

            # Ajuste de confianza si ya hay skill
            if pkey in self.skills:
                sk = self.skills[pkey]
                if accepted:
                    sk.confianza = min(1.0, sk.confianza + 0.05)
                else:
                    sk.confianza = max(0.1, sk.confianza - 0.10)

            # Disparo de codegen
            should_generate = (
                pat.frecuencia >= CODEGEN_THRESHOLD
                and pkey not in self.skills
                and pkey not in self._codegen_in_flight
            )
            if should_generate:
                self._codegen_in_flight.add(pkey)

            self._save_index()

        if should_generate:
            asyncio.create_task(self._generate_skill(pkey, keys, list(pat.examples)))

    async def find_skill(self, query: str) -> Skill | None:
        """Busca skill cuyo score (overlap * confianza) supere 0.6."""
        keys = set(_extract_keywords(query))
        if not keys:
            return None

        best: tuple[float, Skill] | None = None
        for sk in self.skills.values():
            sk_keys = set(sk.keys)
            if not sk_keys:
                continue
            overlap = len(keys & sk_keys) / max(1, len(keys | sk_keys))
            score = overlap * sk.confianza
            if score > 0.6 and (best is None or score > best[0]):
                best = (score, sk)
        return best[1] if best else None

    async def run_skill(
        self,
        skill: Skill,
        query: str,
        datos: list | dict,
    ) -> dict | None:
        """Carga el .py con importlib y ejecuta run(query, datos)."""
        path = SKILLS_DIR / skill.file
        if not path.exists():
            print(f"[SkillStore] skill file no encontrado: {path}")
            return None

        def _exec() -> dict | None:
            try:
                spec = importlib.util.spec_from_file_location(
                    f"skill_{skill.pkey}", str(path)
                )
                if not spec or not spec.loader:
                    return None
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "run"):
                    return None
                return mod.run(query=query, datos=datos)
            except Exception as e:
                print(f"[SkillStore] error ejecutando skill {skill.pkey}: {e}")
                return None

        result = await asyncio.to_thread(_exec)

        async with self._lock:
            skill.veces_usado += 1
            if result and isinstance(result, dict) and "metricas" in result:
                skill.veces_exitoso += 1
                skill.confianza = min(1.0, skill.confianza + 0.02)
            else:
                skill.confianza = max(0.1, skill.confianza - 0.10)
            self._save_index()

        return result if (result and isinstance(result, dict) and "metricas" in result) else None

    # ─── codegen ──────────────────────────────────────────────────────────

    async def _generate_skill(self, pkey: str, keys: list[str], examples: list[str]) -> None:
        """Llama a Gemini CODE para generar un skill. Solo se invoca al llegar al umbral."""
        try:
            print(f"[SkillStore] generando skill para pkey={pkey} keys={keys}")
            prompt = CODEGEN_PROMPT.format(
                keys=", ".join(keys),
                examples="\n".join(f"- {e}" for e in examples[-3:]),
            )
            raw = await gemini.codegen(prompt)
            code = _strip_python_block(raw)
            if not code:
                print(f"[SkillStore] no se pudo extraer código del bloque: {raw[:200]}")
                return

            # Validar sintaxis antes de guardar
            try:
                ast.parse(code)
            except SyntaxError as e:
                print(f"[SkillStore] código inválido (SyntaxError): {e}")
                return

            if "def run" not in code:
                print("[SkillStore] el código no define run() — descartado")
                return

            file_name = f"skill_{pkey}.py"
            path = SKILLS_DIR / file_name
            path.write_text(code, encoding="utf-8")

            async with self._lock:
                self.skills[pkey] = Skill(pkey=pkey, keys=keys, file=file_name)
                self._save_index()

            print(f"[SkillStore] ✓ skill generado: {file_name}")
        except Exception as e:
            print(f"[SkillStore] Error en codegen: {e}")
        finally:
            self._codegen_in_flight.discard(pkey)

    # ─── inspección ───────────────────────────────────────────────────────

    def list_skills(self) -> list[dict]:
        salida = []
        for s in self.skills.values():
            d = asdict(s)
            d["frecuencia_patron"] = self.patterns.get(s.pkey, Pattern(s.pkey, s.keys)).frecuencia
            salida.append(d)
        salida.sort(key=lambda x: x.get("veces_usado", 0), reverse=True)
        return salida

    def list_patterns(self) -> list[dict]:
        salida = [asdict(p) for p in self.patterns.values()]
        salida.sort(key=lambda x: x["frecuencia"], reverse=True)
        return salida


# Singleton
skill_store = SkillStore()
