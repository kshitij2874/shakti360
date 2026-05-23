# 🪷 ShaktiAgent — Multi-Agent AI Life Companion for Women

> **DeployFest 2026** by GDG Cloud Bengaluru  
> GCP Project: `deployfest-kv-2026` | Region: `asia-south1`

ShaktiAgent is a multi-agent AI system that serves as a life-stage companion for women across **3 life pillars** (Health, Finance, Career) and **3 age bands** (11–24, 25–40, 41+). Built with Google ADK, Vertex AI, Firestore, and Langfuse observability.

---

## 🏗️ Architecture

```
User → FastAPI → Orchestrator (classify → route → validate)
                      ├── Health Agent (MedGemma / gemini-2.0-flash)
                      ├── Finance Agent (gemini-2.0-flash)
                      └── Career Agent (gemini-2.0-flash)
                            ↕
                    RAG Pipeline (Firestore vectors)
                    Memory Layer (3-tier: short/episodic/semantic)
                    Human Oversight Gate (approve/reject tool calls)
                    Langfuse Observability
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd shakti360
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your keys:
#   PROJECT_ID, MAPS_API_KEY, LANGFUSE keys (optional)
```

### 3. Run Locally
```bash
python main.py
# or
uvicorn main:app --reload --port 8080
```

Open **http://localhost:8080** for the frontend.

### 4. Deploy to Cloud Run
```bash
chmod +x deploy.sh
./deploy.sh
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Main chat — classify, route, validate |
| `POST` | `/approve` | Approve queued tool calls |
| `POST` | `/reject` | Reject queued tool calls |
| `GET` | `/sessions/{user_id}` | Last 20 sessions |
| `GET` | `/metrics` | Dashboard metrics |
| `GET` | `/health` | Health check |

### Chat Request
```json
{
  "user_id": "user_abc",
  "age_band": "25-40",
  "query": "What is Janani Suraksha Yojana?",
  "session_id": "optional"
}
```

## 📁 Project Structure

```
shakti360/
├── main.py              # FastAPI app, all endpoints
├── orchestrator.py      # Root agent: classify → route → validate
├── agents/
│   ├── health.py        # Health sub-agent
│   ├── finance.py       # Finance sub-agent
│   └── career.py        # Career sub-agent
├── rag.py               # PageIndex-style RAG pipeline
├── memory.py            # 3-tier memory (Tencent pattern)
├── tools.py             # Maps, schemes, SIP calc, helplines
├── validators.py        # Citation, age, safety checks
├── observability.py     # Langfuse + metrics
├── cache.py             # Response caching
├── docs/                # 9 seed RAG documents
├── index.html           # Frontend (React + Tailwind CDN)
├── Dockerfile
├── deploy.sh
├── requirements.txt
└── .env.example
```

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PROJECT_ID` | Yes | GCP project ID |
| `LOCATION` | No | Firestore region (default: asia-south1) |
| `VERTEX_LOCATION` | No | Vertex AI region (default: us-central1) |
| `MAPS_API_KEY` | No | Google Maps API key (mock if missing) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `LANGFUSE_HOST` | No | Langfuse host URL |
| `USE_MEDGEMMA` | No | Enable MedGemma for health agent |

## 🛡️ Safety & Governance

- **No medical diagnosis** — Health agent navigates, never diagnoses
- **No financial guarantees** — Finance agent never promises returns
- **Human oversight gate** — All external tool calls require approval
- **3-layer validation** on every response (citation, age, safety)
- **Fallback to helplines** (181, 1091, 112) when checks fail

## 📊 Observability

All agent functions are instrumented with `@observe()`:
- Traces every classify → route → validate cycle
- Custom metadata: age_band, pillar, citations_count, latency_ms
- Gracefully degrades to console logging without Langfuse keys

---

Built with ❤️ for DeployFest 2026 | GDG Cloud Bengaluru
