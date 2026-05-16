# CLAUDE.md — StringBuilder Analytics

Este archivo le da contexto completo a Claude Code sobre el proyecto.
Léelo completo antes de tocar cualquier archivo.

---

## ¿Qué es este proyecto?

**StringBuilder Analytics** es un sistema de análisis de datos impulsado por IA.
El usuario conecta datos (JSON, CSV, Supabase, API), hace preguntas en lenguaje natural,
y el sistema genera gráficas, métricas, proyecciones e interpretaciones automáticamente.

Con el tiempo, el sistema **aprende patrones del usuario** y genera código Python optimizado
(skills) para responder sin llamar a la API de Gemini — haciéndose más rápido con cada uso.

---

## Stack tecnológico

| Capa | Tecnología | Notas |
|---|---|---|
| Backend | Python 3.12 + FastAPI | REST + WebSocket + SSE |
| IA | Google Gemini API | 3 modelos según complejidad |
| Datos | pandas + scikit-learn | proyecciones, anomalías |
| Chat store | TinyDB (MVP) | migrar a SQLite + Redis en Fase 2 |
| Skills | importlib dinámico | código generado por Gemini Pro |
| Frontend | HTML + Chart.js | sin framework, vanilla JS |
| Deploy | Docker + docker-compose | Redis incluido |

---

## Estructura de archivos

```
stringbuilder-analytics/
├── CLAUDE.md                        ← estás aquí
├── .env                             ← GEMINI_API_KEY (nunca al repo)
├── .env.example                     ← plantilla pública
├── .gitignore
├── docker-compose.yml               ← backend + redis
├── README.md
│
├── backend/
│   ├── main.py                      ← FastAPI app principal — PUNTO DE ENTRADA
│   ├── Dockerfile
│   ├── requirements.txt
│   └── core/
│       ├── __init__.py
│       ├── gemini.py                ← cliente Gemini, routing por modelo
│       ├── chat_store.py            ← sesiones y mensajes (TinyDB)
│       ├── skills.py                ← skill engine + codegen
│       └── projections.py          ← regresión lineal, detección anomalías
│
├── frontend/
│   └── index.html                   ← dashboard completo (vanilla JS)
│
├── data/                            ← TinyDB sessions.json (gitignored)
└── skills/
    └── generated/                   ← .py generados por codegen (gitignored)
```

---

## Cómo correr el proyecto

### Con Docker (recomendado)

```bash
cp .env.example .env
# edita .env y pon tu GEMINI_API_KEY

docker compose up --build
# http://localhost:8080
```

### Sin Docker (desarrollo local)

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=AIza_tu_key
uvicorn main:app --reload --port 8080
```

---

## Variables de entorno requeridas

```bash
GEMINI_API_KEY=AIza...   # OBLIGATORIO — https://aistudio.google.com/apikey
PORT=8080                # opcional, default 8080
LOG_LEVEL=info           # debug | info | warning | error
```

---

## API endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/query` | Análisis completo: datos + pregunta → métricas + gráficas |
| `WS` | `/api/chat` | Chat conversacional con memoria de sesión |
| `GET` | `/api/stream` | SSE streaming token a token |
| `POST` | `/api/projection` | Proyección lineal de serie numérica |
| `POST` | `/api/anomalias` | Detección de outliers con Z-score |
| `GET` | `/api/skills` | Lista skills aprendidos y sus métricas |
| `POST` | `/api/sessions` | Nueva sesión de chat |
| `GET` | `/api/sessions` | Lista sesiones del usuario |
| `GET` | `/api/sessions/{a}/diff/{b}` | Compara dos sesiones |
| `GET` | `/api/health` | Estado del sistema |

### Ejemplo de POST /api/query

```json
// Request
{
  "pregunta": "¿Cuál fue el mes con más ventas?",
  "datos": [
    {"mes": "Enero", "ventas": 45000, "gastos": 28000},
    {"mes": "Febrero", "ventas": 52000, "gastos": 31000}
  ],
  "session_id": "uuid-opcional"
}

// Response
{
  "metricas": [{"label": "Mejor mes", "valor": "Febrero"}],
  "interpretacion": "Febrero registró las mayores ventas...",
  "graficas": [{
    "titulo": "Ventas por mes",
    "tipo": "bar",
    "datos": {
      "labels": ["Enero", "Febrero"],
      "datasets": [{"label": "Ventas", "data": [45000, 52000]}]
    }
  }],
  "skill_usado": false
}
```

---

## Módulos principales — lo que necesitas saber antes de editar

### `backend/core/gemini.py`

Contiene el `GeminiClient` con tres modelos:

```python
class Modelo(str, Enum):
    FAST = "gemini-2.5-flash-lite-preview-06-17"  # 30 RPM — clasificar
    MAIN = "gemini-2.5-flash"                      # 15 RPM — análisis (principal)
    CODE = "gemini-2.5-pro"                        # 5 RPM  — codegen skills
```

- `gemini.ask(prompt, Modelo.MAIN)` → str
- `gemini.ask_stream(prompt, Modelo.MAIN)` → AsyncGenerator (para SSE)
- `gemini.analyze(pregunta, datos)` → `QueryResponse` con métricas + gráficas

**Regla de uso de modelos:** FAST para clasificación/routing, MAIN para todo lo normal,
CODE solo cuando el `SkillStore._generate_skill()` necesita generar código nuevo.
Las 5 RPD de CODE son preciosas — no desperdiciarlas.

### `backend/core/skills.py`

El `SkillStore` aprende de cada query:

1. `skill_store.find_skill(query)` → busca skill con confianza > 0.6
2. `skill_store.run_skill(skill, query, datos)` → ejecuta el .py generado
3. `skill_store.observe(query, response)` → acumula frecuencia de patrones
4. Cuando un patrón llega a `CODEGEN_THRESHOLD = 3` repeticiones → llama a Gemini CODE

Los skills generados se guardan en `skills/generated/skill_HASH.py`.
Cada skill es un módulo Python con una función `run(query, datos) -> dict`.

**No borrar** la carpeta `skills/generated/` — ahí viven los skills aprendidos.

### `backend/core/chat_store.py`

Persiste sesiones en `data/sessions.json` (TinyDB).

- `chat_store.new_session()` → UUID de la sesión
- `chat_store.save_message(session_id, role, content)` → guarda mensaje
- `chat_store.get_context_str(session_id)` → historial como string para prompts
- `chat_store.diff_sessions(id_a, id_b)` → compara dos sesiones

**Fase 2:** reemplazar TinyDB por SQLite (`sqlx` en Rust o `aiosqlite` en Python)
y Redis para las sesiones activas. El interface público no cambia.

### `backend/core/projections.py`

Funciones puras de ML — sin llamadas a API:

- `linear_forecast(valores, periodos)` → proyección + config Chart.js lista
- `detect_anomalies(valores, umbral_sigma)` → outliers con Z-score

Estas dos funciones son candidatas a migrar a Rust en Fase 3.

### `backend/main.py`

FastAPI app. Registra los tres protocolos de comunicación:

```python
# ① REST
@app.post("/api/query")

# ② WebSocket
@app.websocket("/api/chat")

# ③ SSE streaming
@app.get("/api/stream")
```

El frontend estático en `/frontend/index.html` se sirve desde aquí también.
CORS está en modo `allow_origins=["*"]` — cambiar antes de producción.

---

## Protocolos de comunicación

### ① REST — query única
```
Frontend → POST /api/query → FastAPI → Gemini → JSON response → Frontend
```
Flujo síncrono. El frontend espera la respuesta completa antes de renderizar.

### ② WebSocket — chat persistente
```
Frontend ←→ WS /api/chat ←→ FastAPI (guarda en TinyDB) ←→ Gemini
```
Conexión persistente. Cada mensaje se guarda automáticamente en la sesión.
El contexto de la conversación se incluye en cada prompt a Gemini.

### ③ SSE — streaming token a token
```
Frontend ← EventSource /api/stream ← FastAPI ← Gemini (stream=True)
```
Gemini devuelve chunks. FastAPI los reenvía inmediatamente.
El frontend acumula tokens y los muestra letra a letra (efecto Jarvis).

---

## Convenciones de código

- **Python:** type hints en todas las funciones, dataclasses para modelos internos, Pydantic para modelos de API
- **Async:** usar `asyncio.to_thread()` para llamadas síncronas bloqueantes (Gemini SDK)
- **Errores:** capturar siempre con try/except en endpoints, nunca dejar que un fallo de Gemini rompa el servidor
- **Logs:** usar `print(f"[Modulo] mensaje")` en desarrollo, migrar a `logging` en producción
- **Skills generados:** siempre validar que el código generado por Gemini compila antes de guardarlo

---

## Roadmap de desarrollo

### Fase 1 — MVP actual ✅
- [x] REST + WebSocket + SSE
- [x] Gemini con routing por modelo
- [x] Skill engine básico (keywords + frecuencia)
- [x] Proyecciones con scikit-learn
- [x] Chat store con TinyDB
- [x] Frontend dashboard dark mode
- [x] Docker + docker-compose

### Fase 2 — Siguiente paso
- [ ] SQLite con `aiosqlite` para chat store persistente
- [ ] Redis para caché de sesiones activas
- [ ] Skill trigger semántico con embeddings (`sentence-transformers`)
- [ ] Auth con JWT (agregar middleware en `main.py`)
- [ ] Tests con `pytest` + `httpx`
- [ ] Rate limiting por usuario en endpoints de Gemini

### Fase 3 — Escala
- [ ] Migrar `projections.py` a Rust (puro cómputo, sin IA)
- [ ] Migrar `chat_store` a LMDB en Rust (zero-copy reads)
- [ ] Skill hot-swap con `libloading` en Rust
- [ ] Python y Rust conviven vía REST interno

---

## Errores comunes y soluciones

### `GEMINI_API_KEY no configurada`
```bash
cp .env.example .env
# edita .env con tu key
```

### `ModuleNotFoundError: No module named 'google.generativeai'`
```bash
pip install -r backend/requirements.txt
```

### Skills no se generan
Verificar que `CODEGEN_THRESHOLD = 3` en `skills.py` — necesitas repetir el mismo tipo de query 3 veces.
Revisar los logs del servidor por `[SkillStore] Error en codegen`.

### Frontend no carga
El frontend se sirve desde `frontend/index.html`. Si corres el backend desde `backend/`,
el path relativo `frontend/` no existe. Correr desde la raíz del proyecto o ajustar el path en `main.py`.

### WebSocket se desconecta inmediatamente
Revisar que el frontend use `ws://` para HTTP y `wss://` para HTTPS.
El código del frontend lo detecta automáticamente con `location.protocol`.

---

## Notas para Claude Code

- El archivo de entrada es `backend/main.py`
- Para agregar un endpoint nuevo: agregar el handler y registrarlo en `main.py`
- Para extender el skill engine: editar `backend/core/skills.py`
- Para cambiar modelos de Gemini: editar el `Enum Modelo` en `backend/core/gemini.py`
- Los datos de TinyDB se guardan en `data/sessions.json` — no tocar manualmente
- Los skills generados en `skills/generated/` son código autogenerado — no editar a mano
- Si algo falla en un endpoint, siempre revisar el health check primero: `GET /api/health`
- El frontend usa `const API = ''` — mismo origen. Si el backend corre en otro puerto, cambiar a `const API = 'http://localhost:8080'`
