"""Cliente Gemini con routing por modelo según complejidad.

3 modelos:
- FAST: clasificación / routing (30 RPM)
- MAIN: análisis principal y chat (15 RPM)
- CODE: codegen de skills (5 RPD - usar con cuidado)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator

import google.generativeai as genai


class Modelo(str, Enum):
    FAST = "gemini-2.5-flash-lite-preview-06-17"
    MAIN = "gemini-2.5-flash"
    CODE = "gemini-2.5-pro"


SYSTEM_PROMPT_ANALISIS = """Eres el cerebro analítico de StringBuilder Analytics.
Recibes datos JSON y una pregunta del usuario.
Responde ÚNICAMENTE con JSON válido — sin backticks, sin texto extra.
Esquema:
{
  "metricas": [{"label": "string", "valor": "string"}],
  "interpretacion": "2-3 oraciones en español",
  "graficas": [{
    "titulo": "string",
    "tipo": "bar|line|pie|doughnut|horizontal_bar",
    "datos": {
      "labels": ["string"],
      "datasets": [{"label": "string", "data": [number]}]
    }
  }]
}
Genera 3-4 métricas KPI y 2-3 gráficas relevantes.
Elige el tipo de gráfica más apropiado para cada análisis."""


SYSTEM_PROMPT_CHAT = """Eres StringBuilder Analytics, un asistente analítico estilo JARVIS.
Respondes en español, con tono profesional pero cercano.
Si el usuario te da datos, los analizas. Si no, conversas normalmente.
Mantén el contexto de la conversación previa."""


@dataclass
class GeminiClient:
    """Cliente único para los 3 modelos de Gemini."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    _models: dict[Modelo, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_key:
            print("[Gemini] ⚠ GEMINI_API_KEY no configurada — el cliente fallará en cada llamada")
            return
        genai.configure(api_key=self.api_key)
        for modelo in Modelo:
            try:
                self._models[modelo] = genai.GenerativeModel(modelo.value)
                print(f"[Gemini] modelo cargado: {modelo.name} → {modelo.value}")
            except Exception as e:
                print(f"[Gemini] error cargando {modelo.name}: {e}")

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _get_model(self, modelo: Modelo) -> Any:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY no configurada")
        if modelo not in self._models:
            self._models[modelo] = genai.GenerativeModel(modelo.value)
        return self._models[modelo]

    @staticmethod
    def _strip_json(text: str) -> str:
        """Limpia respuestas de Gemini que vienen con backticks o texto extra."""
        text = text.strip()
        m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return m.group(1)
        return text

    # ─── API pública ──────────────────────────────────────────────────────

    async def ask(
        self,
        prompt: str,
        modelo: Modelo = Modelo.MAIN,
        system: str | None = None,
    ) -> str:
        """Llamada simple. Devuelve string crudo."""
        model = self._get_model(modelo)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        def _call() -> str:
            try:
                resp = model.generate_content(full_prompt)
                return resp.text or ""
            except Exception as e:
                print(f"[Gemini.ask] error en {modelo.name}: {e}")
                raise

        return await asyncio.to_thread(_call)

    async def ask_stream(
        self,
        prompt: str,
        modelo: Modelo = Modelo.MAIN,
        system: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming token a token vía generate_content(stream=True)."""
        model = self._get_model(modelo)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        def _start_stream() -> Any:
            return model.generate_content(full_prompt, stream=True)

        try:
            stream = await asyncio.to_thread(_start_stream)

            def _next_chunk(it: Any) -> str | None:
                try:
                    chunk = next(it)
                    return chunk.text or ""
                except StopIteration:
                    return None
                except Exception as e:
                    print(f"[Gemini.ask_stream] error en chunk: {e}")
                    return None

            iterator = iter(stream)
            while True:
                text = await asyncio.to_thread(_next_chunk, iterator)
                if text is None:
                    break
                if text:
                    yield text
        except Exception as e:
            print(f"[Gemini.ask_stream] error iniciando stream: {e}")
            yield f"[error] {e}"

    async def analyze(
        self,
        pregunta: str,
        datos: list | dict,
        contexto: str | None = None,
    ) -> dict:
        """Analiza datos y devuelve dict con metricas/interpretacion/graficas."""
        ctx = f"\nContexto previo de la conversación:\n{contexto}\n" if contexto else ""
        datos_json = json.dumps(datos, ensure_ascii=False, default=str)
        if len(datos_json) > 20000:
            datos_json = datos_json[:20000] + "...[truncado]"

        prompt = f"""{ctx}
Pregunta del usuario: {pregunta}

Datos:
{datos_json}

Devuelve ÚNICAMENTE el JSON con el esquema indicado."""

        raw = await self.ask(prompt, Modelo.MAIN, system=SYSTEM_PROMPT_ANALISIS)
        cleaned = self._strip_json(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[Gemini.analyze] JSON inválido: {e}\nRaw: {raw[:500]}")
            return {
                "metricas": [{"label": "Error", "valor": "Respuesta no JSON"}],
                "interpretacion": f"El modelo devolvió texto inválido. Crudo: {raw[:300]}",
                "graficas": [],
            }

    async def chat(self, mensaje: str, contexto: str | None = None) -> str:
        """Chat conversacional con contexto opcional."""
        ctx = f"\nHistorial:\n{contexto}\n" if contexto else ""
        prompt = f"{ctx}\nUsuario: {mensaje}\nAsistente:"
        return await self.ask(prompt, Modelo.MAIN, system=SYSTEM_PROMPT_CHAT)

    async def classify(self, query: str) -> str:
        """Routing: clasifica una query como 'simple' o 'compleja'. Usa FAST."""
        prompt = f"""Clasifica esta query como 'simple' o 'compleja'.
'simple' = saludo, pregunta corta sin datos, conversación.
'compleja' = análisis de datos, gráficas, métricas, proyecciones.

Query: {query}

Responde solo una palabra: simple o compleja."""
        try:
            r = await self.ask(prompt, Modelo.FAST)
            return "compleja" if "compl" in r.lower() else "simple"
        except Exception:
            return "compleja"

    async def codegen(self, prompt: str) -> str:
        """Genera código Python con el modelo CODE. Usar con moderación."""
        return await self.ask(prompt, Modelo.CODE)


# Instancia singleton compartida por todos los módulos
gemini = GeminiClient()
