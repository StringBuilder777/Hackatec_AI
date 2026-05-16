"""FastAPI app principal de StringBuilder Analytics.

Tres protocolos:
  ① REST          → POST /api/query
  ② WebSocket     → WS   /api/chat
  ③ SSE streaming → GET  /api/stream
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.gemini import gemini, Modelo, SYSTEM_PROMPT_CHAT
from core.chat_store import chat_store
from core.skills import skill_store
from core.projections import linear_forecast, detect_anomalies


# ─── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StringBuilder Analytics",
    description="Sistema de análisis de datos con IA — REST + WebSocket + SSE",
    version="0.1.0",
)

# CORS permissive en desarrollo. Restringir antes de producción.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Modelos Pydantic ─────────────────────────────────────────────────────

class Metrica(BaseModel):
    label: str
    valor: str


class GraficaDataset(BaseModel):
    label: str
    data: list[float | None]


class GraficaDatos(BaseModel):
    labels: list[str]
    datasets: list[GraficaDataset]


class GraficaConfig(BaseModel):
    titulo: str
    tipo: str
    datos: GraficaDatos


class QueryRequest(BaseModel):
    session_id: str | None = None
    pregunta: str
    datos: list | dict


class QueryResponse(BaseModel):
    metricas: list[Metrica] = []
    interpretacion: str = ""
    graficas: list[GraficaConfig] = []
    skill_usado: bool = False


class ProjectionRequest(BaseModel):
    valores: list[float]
    periodos: int = 6


class AnomaliasRequest(BaseModel):
    valores: list[float]
    umbral_sigma: float = 2.0


# ─── Endpoints REST ───────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest) -> QueryResponse:
    """Análisis completo. Intenta usar un skill aprendido antes de llamar a Gemini."""
    try:
        # 1. Buscar skill aprendido
        skill = await skill_store.find_skill(req.pregunta)
        result: dict | None = None
        skill_usado = False

        if skill:
            print(f"[/api/query] usando skill {skill.file} (conf={skill.confianza:.2f})")
            result = await skill_store.run_skill(skill, req.pregunta, req.datos)
            if result:
                skill_usado = True

        # 2. Si no hay skill o el skill falló, llamar a Gemini
        if result is None:
            contexto = ""
            if req.session_id:
                contexto = await chat_store.get_context_str(req.session_id)
            result = await gemini.analyze(
                pregunta=req.pregunta,
                datos=req.datos,
                contexto=contexto or None,
            )

        # 3. Persistir en sesión + observar para skill engine
        if req.session_id:
            await chat_store.save_message(req.session_id, "user", req.pregunta)
            await chat_store.save_message(
                req.session_id, "assistant",
                result.get("interpretacion", ""),
            )
            await chat_store.set_datos_hash(req.session_id, req.datos)

        await skill_store.observe(req.pregunta, result, accepted=True)

        return QueryResponse(
            metricas=[Metrica(**m) for m in result.get("metricas", [])],
            interpretacion=result.get("interpretacion", ""),
            graficas=[GraficaConfig(**g) for g in result.get("graficas", [])],
            skill_usado=skill_usado,
        )
    except Exception as e:
        print(f"[/api/query] ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {e}")


@app.post("/api/projection")
async def api_projection(req: ProjectionRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(linear_forecast, req.valores, req.periodos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/anomalias")
async def api_anomalias(req: AnomaliasRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(detect_anomalies, req.valores, req.umbral_sigma)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/skills")
async def api_skills() -> dict[str, Any]:
    return {
        "skills": skill_store.list_skills(),
        "patterns": skill_store.list_patterns(),
        "total_skills": len(skill_store.skills),
        "total_patterns": len(skill_store.patterns),
    }


@app.post("/api/sessions")
async def api_new_session(payload: dict | None = None) -> dict[str, Any]:
    titulo = (payload or {}).get("titulo", "Nueva sesión")
    sid = await chat_store.new_session(titulo)
    return {"session_id": sid, "titulo": titulo}


@app.get("/api/sessions")
async def api_list_sessions() -> dict[str, Any]:
    return {"sesiones": await chat_store.list_sessions()}


@app.get("/api/sessions/{a}/diff/{b}")
async def api_diff_sessions(a: str, b: str) -> dict[str, Any]:
    return await chat_store.diff_sessions(a, b)


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "gemini_configured": bool(gemini.api_key),
        "modelos": {m.name: m.value for m in Modelo},
        "skills_aprendidos": len(skill_store.skills),
        "patrones_observados": len(skill_store.patterns),
        "ts": time.time(),
    }


# ─── WebSocket /api/chat ──────────────────────────────────────────────────

@app.websocket("/api/chat")
async def ws_chat(ws: WebSocket) -> None:
    await ws.accept()
    sid = await chat_store.new_session("Chat WS")
    await ws.send_json({"type": "session", "session_id": sid})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"content": raw}

            content = msg.get("content", "").strip()
            if not content:
                continue

            await chat_store.save_message(sid, "user", content)
            contexto = await chat_store.get_context_str(sid)

            try:
                respuesta = await gemini.chat(content, contexto=contexto)
            except Exception as e:
                respuesta = f"[error] {e}"

            await chat_store.save_message(sid, "assistant", respuesta)
            await ws.send_json({
                "type": "message",
                "role": "assistant",
                "content": respuesta,
                "ts": time.time(),
            })
    except WebSocketDisconnect:
        print(f"[ws] sesión {sid} desconectada")
    except Exception as e:
        print(f"[ws] error: {e}")
        try:
            await ws.close()
        except Exception:
            pass


# ─── SSE /api/stream ──────────────────────────────────────────────────────

@app.get("/api/stream")
async def api_stream(pregunta: str = Query(...)) -> StreamingResponse:
    """SSE: devuelve tokens del modelo MAIN a medida que llegan."""

    async def event_gen():
        try:
            yield f"event: start\ndata: {json.dumps({'pregunta': pregunta})}\n\n"
            async for chunk in gemini.ask_stream(
                pregunta, Modelo.MAIN, system=SYSTEM_PROMPT_CHAT
            ):
                payload = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"event: token\ndata: {payload}\n\n"
            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Static frontend ──────────────────────────────────────────────────────

# Sirve frontend/index.html. La ruta es relativa al CWD de uvicorn.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if not FRONTEND_DIR.exists():
    # Fallback: cuando docker monta /app/frontend
    FRONTEND_DIR = Path("/app/frontend")


@app.get("/")
async def index() -> FileResponse:
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse(
        {"error": f"frontend no encontrado en {FRONTEND_DIR}"},
        status_code=404,
    )


# Sirve assets estáticos adicionales si existen
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Entry point local ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
