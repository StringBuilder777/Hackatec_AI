"""Chat store con TinyDB para sesiones y mensajes.

Cada sesión guarda historial completo + hash de los datos para detectar cambios.
TinyDB no es thread-safe, así que envolvemos toda lectura/escritura en asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from tinydb import TinyDB, Query


DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sessions.json"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Mensaje:
    role: str        # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Sesion:
    id: str
    titulo: str
    messages: list[dict] = field(default_factory=list)
    proyecciones: list[dict] = field(default_factory=list)
    datos_hash: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1


def _hash_datos(datos: Any) -> str:
    if datos is None:
        return ""
    blob = json.dumps(datos, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ChatStore:
    """Wrapper async sobre TinyDB."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db = TinyDB(str(db_path), indent=2, ensure_ascii=False)
        self.tabla = self.db.table("sesiones")
        self._lock = asyncio.Lock()

    # ─── Sync helpers (usados desde to_thread) ────────────────────────────

    def _new_session_sync(self, titulo: str = "Nueva sesión") -> str:
        sid = str(uuid.uuid4())
        s = Sesion(id=sid, titulo=titulo)
        self.tabla.insert(asdict(s))
        return sid

    def _get_session_sync(self, sid: str) -> dict | None:
        Q = Query()
        results = self.tabla.search(Q.id == sid)
        return results[0] if results else None

    def _save_message_sync(self, sid: str, role: str, content: str) -> None:
        Q = Query()
        s = self._get_session_sync(sid)
        if not s:
            return
        msg = asdict(Mensaje(role=role, content=content))
        s["messages"].append(msg)
        s["updated_at"] = time.time()
        s["version"] = s.get("version", 1) + 1
        # Actualizar título si es el primer mensaje del usuario
        if role == "user" and len(s["messages"]) == 1:
            s["titulo"] = (content[:50] + "...") if len(content) > 50 else content
        self.tabla.update(s, Q.id == sid)

    def _set_datos_hash_sync(self, sid: str, datos: Any) -> None:
        Q = Query()
        s = self._get_session_sync(sid)
        if not s:
            return
        s["datos_hash"] = _hash_datos(datos)
        s["updated_at"] = time.time()
        s["version"] = s.get("version", 1) + 1
        self.tabla.update(s, Q.id == sid)

    def _list_sessions_sync(self) -> list[dict]:
        sesiones = self.tabla.all()
        # Resumen ordenado por updated_at desc
        resumen = [
            {
                "id": s["id"],
                "titulo": s.get("titulo", ""),
                "n_mensajes": len(s.get("messages", [])),
                "updated_at": s.get("updated_at", 0),
                "version": s.get("version", 1),
            }
            for s in sesiones
        ]
        resumen.sort(key=lambda x: x["updated_at"], reverse=True)
        return resumen

    def _diff_sessions_sync(self, id_a: str, id_b: str) -> dict:
        a = self._get_session_sync(id_a) or {}
        b = self._get_session_sync(id_b) or {}
        msgs_a = a.get("messages", [])
        msgs_b = b.get("messages", [])
        return {
            "mensajes_a": len(msgs_a),
            "mensajes_b": len(msgs_b),
            "diff_mensajes": len(msgs_b) - len(msgs_a),
            "datos_cambiaron": a.get("datos_hash", "") != b.get("datos_hash", ""),
            "versions": {"a": a.get("version", 0), "b": b.get("version", 0)},
        }

    # ─── API pública async ────────────────────────────────────────────────

    async def new_session(self, titulo: str = "Nueva sesión") -> str:
        async with self._lock:
            return await asyncio.to_thread(self._new_session_sync, titulo)

    async def get_session(self, sid: str) -> dict | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_session_sync, sid)

    async def save_message(self, sid: str, role: str, content: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_message_sync, sid, role, content)

    async def set_datos_hash(self, sid: str, datos: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self._set_datos_hash_sync, sid, datos)

    async def list_sessions(self) -> list[dict]:
        async with self._lock:
            return await asyncio.to_thread(self._list_sessions_sync)

    async def diff_sessions(self, id_a: str, id_b: str) -> dict:
        async with self._lock:
            return await asyncio.to_thread(self._diff_sessions_sync, id_a, id_b)

    async def get_context_str(self, sid: str, max_msgs: int = 10) -> str:
        """Devuelve los últimos max_msgs como string para incluir en prompts."""
        s = await self.get_session(sid)
        if not s:
            return ""
        msgs = s.get("messages", [])[-max_msgs:]
        lines = []
        for m in msgs:
            tag = "Usuario" if m["role"] == "user" else "Asistente"
            lines.append(f"{tag}: {m['content']}")
        return "\n".join(lines)


# Singleton compartido
chat_store = ChatStore()
