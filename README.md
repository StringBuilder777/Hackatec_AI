# StringBuilder Analytics

Sistema de análisis de datos con IA inspirado en JARVIS.
Conecta datos (JSON, CSV, APIs), pregunta en lenguaje natural y obtén
gráficas, métricas, proyecciones e interpretaciones automáticas.

Con el tiempo, el sistema **aprende** patrones del usuario y genera código
Python optimizado (skills) que responde sin llamar a la API de Gemini.

---

## Quick start (Docker)

```bash
cp .env.example .env
# editar .env y poner tu GEMINI_API_KEY
# (obtener en https://aistudio.google.com/apikey)

docker compose up --build
```

Abre http://localhost:8080

---

## Quick start (local sin Docker)

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=AIza_tu_key
uvicorn main:app --reload --port 8080
```

Abre http://localhost:8080

---

## Stack

| Capa | Tech |
|---|---|
| Backend | Python 3.12 + FastAPI |
| IA | Google Gemini (3 modelos: FAST/MAIN/CODE) |
| Datos | pandas + scikit-learn |
| Chat store | TinyDB (MVP) |
| Skills | importlib dinámico |
| Frontend | HTML + Chart.js vanilla |
| Deploy | Docker + docker-compose |

---

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/query` | Análisis: datos + pregunta → métricas + gráficas |
| `WS` | `/api/chat` | Chat persistente con sesión |
| `GET` | `/api/stream?pregunta=...` | SSE streaming token a token |
| `POST` | `/api/projection` | Proyección lineal `{valores, periodos}` |
| `POST` | `/api/anomalias` | Outliers `{valores, umbral_sigma}` |
| `GET` | `/api/skills` | Skills aprendidos y métricas |
| `POST/GET` | `/api/sessions` | Crear / listar sesiones |
| `GET` | `/api/sessions/{a}/diff/{b}` | Diff entre sesiones |
| `GET` | `/api/health` | Estado del sistema |

---

## Arquitectura

Ver `CLAUDE.md` para detalles completos. Los módulos clave viven en
`backend/core/`:

- `gemini.py` — cliente con routing por modelo
- `chat_store.py` — sesiones TinyDB
- `skills.py` — skill engine con codegen
- `projections.py` — regresión lineal + Z-score

---

## Cómo funciona el aprendizaje

1. Cada query se observa (`SkillStore.observe`).
2. Se hashean los keywords y se cuenta la frecuencia del patrón.
3. Cuando un patrón repite **3 veces** (`CODEGEN_THRESHOLD = 3`), se llama
   a Gemini Pro para generar un módulo Python con `def run(query, datos)`.
4. El módulo se valida con `ast.parse`, se guarda en `skills/generated/`
   y se registra en el índice.
5. Las queries futuras similares usan el skill — sin llamar a Gemini.

---

## Roadmap

- **Fase 1** (actual): MVP con TinyDB + REST/WS/SSE.
- **Fase 2**: SQLite + Redis, embeddings semánticos, JWT, tests.
- **Fase 3**: migrar `projections` y `chat_store` a Rust.
# Hackatec_AI
