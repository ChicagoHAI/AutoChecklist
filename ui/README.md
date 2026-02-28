# AutoChecklist UI

Web interface for the autochecklist library, providing checklist-based LLM evaluation through a browser.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Next.js App   │  HTTP   │  FastAPI Server  │
│   (React/TS)    │ <─────> │    (Python)      │
│   Port 7770     │   API   │    Port 7771     │
└─────────────────┘         └─────────────────┘
                                   │
                                   v
                           ┌─────────────────┐
                           │  autochecklist   │
                           │    (library)     │
                           └─────────────────┘
```

## Prerequisites

- Python 3.10+ with `uv` package manager
- Node.js 18+ with `npm`
- An API key for OpenRouter, OpenAI, or a running vLLM server

## Quick Start

### Production (default)

```bash
cd ui
./launch_ui.sh
```

Builds the frontend and starts both services in production mode.

### Development (with hot-reload)

```bash
cd ui
./launch_ui.sh --dev
```

Starts both services with hot-reload (auto-reloads on Python/TS changes).

### Environment Variables

Set these in the project root `.env` file:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes (if using OpenRouter) | OpenRouter API key |
| `OPENAI_API_KEY` | No | OpenAI API key (for direct OpenAI usage) |
| `NEXT_PUBLIC_API_URL` | No | Backend URL override (default: http://localhost:7771) |

Alternatively, configure API keys through the Settings page in the UI.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Evaluate page — tabs: **Custom Eval**, **Compare**, **Reference** |
| `/batch` | Batch evaluation — upload CSV/JSON, track progress, export results |
| `/build` | Checklist builder — define rubric dimensions, generate checklists via DeductiveGenerator |
| `/library` | Resource library — **Checklists**, **Prompt Templates**, and **Pipelines** tabs |
| `/settings` | API key configuration, default model, connection testing |

- **Frontend state persistence**: The Custom Eval editor persists state to `sessionStorage` — prompts, generator class, scorer, and input fields survive page navigation.
- **Async evaluation flow**: Custom Eval and Compare submit async jobs → all methods/pipelines run **concurrently** on the backend (`asyncio.create_task` + `asyncio.Lock`) → frontend polls `/api/evaluate/{eval_id}` every 1s → partial results render as methods complete.


### Custom Eval Tab

Full prompt editing workflow for single evaluations:
- Choose generator class (Direct or Contrastive)
- Edit generator prompt and scorer prompt in tabbed editors
- Load from saved prompt templates (filtered by generator class)
- Select output format and scorer type
- Run evaluation, view checklist + score results
- Save results as checklist, prompt template, or pipeline

### Compare Tab

Side-by-side comparison of built-in methods and custom pipelines:
- Toggle built-in methods (TICK, RLCF variants, RocketEval)
- Toggle custom pipelines from the library
- Comparison settings (methods, models) in a collapsible expander
- Results displayed as horizontally-scrollable method cards

### Build

Interactive checklist builder using DeductiveGenerator:
- Define rubric dimensions and sub-dimensions
- Choose augmentation mode (seed, elaboration, diversification, combined)
- Generate checklists and save to library

### Library

Three resource types managed under `/library`:
- **Checklists**: Saved checklists — browse, view, edit, import, delete.
- **Prompt Templates**: Reusable prompts with placeholders (`{input}`, `{target}`, `{candidates}`). Types displayed as DirectGenerator, ContrastiveGenerator, or Scorer.
- **Pipelines**: Complete evaluation configurations bundling generator class, generator prompt, scorer type, scorer prompt, and output format.

## Directory Structure

```
ui/
├── launch_ui.sh            # Launcher (--dev for hot-reload)
├── README.md
├── CLAUDE.md               # Claude Code development guidance
│
├── backend/                # FastAPI (Python)
│   ├── app/
│   │   ├── main.py         # FastAPI entry, CORS, lifespan, router registration
│   │   ├── schemas.py      # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── evaluate.py         # Evaluation endpoints (sync, stream, async, generate, score)
│   │   │   ├── batch.py            # Batch upload, start, status, results, export
│   │   │   ├── checklists.py       # Checklist library CRUD + import
│   │   │   ├── prompt_templates.py # Prompt template CRUD + seed defaults
│   │   │   ├── pipelines.py        # Custom pipeline CRUD
│   │   │   ├── registry_info.py    # Available generators/scorers + preset prompts
│   │   │   └── settings.py         # Settings CRUD + connection test
│   │   └── services/
│   │       ├── checklist.py        # Wraps ChecklistPipeline (method + pipeline-based)
│   │       ├── batch_runner.py     # Background batch job execution
│   │       ├── settings_service.py # 3-tier settings resolution (JSON > .env > env vars)
│   │       └── storage.py          # Atomic JSON read/write to data/
│   ├── data/                       # JSON file storage (no database)
│   │   ├── batches/
│   │   ├── checklists/
│   │   ├── evaluations/
│   │   ├── pipelines/
│   │   └── prompt_templates/
│   └── tests/                      # 150+ unit tests (no API key needed)
│
└── frontend/               # Next.js 16 (TypeScript)
    ├── package.json
    └── src/
        ├── app/                          # App router pages
        │   ├── layout.tsx                # Root layout (fonts, providers, TopNav)
        │   ├── page.tsx                  # Evaluate page (Custom Eval / Compare / Reference tabs)
        │   ├── globals.css               # Tailwind + CSS variables (accent #800000)
        │   ├── batch/
        │   │   ├── page.tsx              # Batch list + upload
        │   │   └── [id]/page.tsx         # Batch detail + results
        │   ├── build/
        │   │   └── page.tsx              # Checklist builder (DeductiveGenerator)
        │   ├── checklists/
        │   │   ├── page.tsx              # Checklist library
        │   │   └── [id]/page.tsx         # Checklist detail
        │   ├── library/
        │   │   ├── page.tsx              # Library (Checklists / Prompts / Pipelines tabs)
        │   │   ├── prompts/[id]/page.tsx # Prompt template editor
        │   │   └── pipelines/[id]/page.tsx # Pipeline config editor
        │   └── settings/
        │       └── page.tsx              # Settings page
        ├── components/
        │   ├── TopNav.tsx                # Navigation bar (Evaluate, Batch, Build, Library)
        │   ├── OnboardingDialog.tsx      # First-run API key setup
        │   ├── MethodCard.tsx            # Single method result card
        │   ├── MethodReferenceTable.tsx  # Built-in methods reference
        │   ├── ChecklistDisplay.tsx      # Checklist item renderer
        │   ├── ExampleLoader.tsx         # Load example data
        │   ├── ui/                       # Radix UI-based primitives
        │   │   ├── Button, Card, Input, Textarea, Select
        │   │   ├── Dialog, Tabs, Tooltip, Badge
        │   │   ├── Progress, Score, Skeleton
        │   │   └── (12 total)
        │   ├── layout/
        │   │   ├── PageHeader.tsx        # Page title + description
        │   │   └── ModelSelector.tsx     # Provider + model dropdown
        │   ├── playground/
        │   │   ├── PlaygroundForm.tsx    # Custom Eval form (prompt editors, eval, save)
        │   │   ├── PromptEditor.tsx      # Code editor for prompts
        │   │   └── ScorerConfigPanel.tsx # Scorer mode/metric configuration
        │   ├── compare/
        │   │   └── CompareForm.tsx       # Method comparison form
        │   ├── checklist_builder/
        │   │   └── DimensionForm.tsx     # Dimension/sub-dimension editor for Build page
        │   ├── batch/
        │   │   ├── BatchUpload.tsx       # File upload
        │   │   ├── BatchProgress.tsx     # Progress bar + status
        │   │   └── BatchResults.tsx      # Results table + pagination
        │   └── library/
        │       ├── PromptTemplateList.tsx # Prompt template table
        │       └── PipelineList.tsx       # Pipeline table
        └── lib/
            ├── types.ts             # TypeScript interfaces + constants
            ├── api.ts               # API client (all backend endpoints)
            ├── hooks.ts             # React Query hooks
            ├── query-provider.tsx   # TanStack Query provider
            └── examples.ts          # Pre-defined evaluation examples
```

## API Endpoints

### Evaluation (`/api/evaluate`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/evaluate/async` | Start async evaluation (returns eval_id for polling) |
| GET | `/api/evaluate/{eval_id}` | Poll async evaluation status and partial results |
| POST | `/api/evaluate/stream` | SSE streaming evaluation (generate + score) |
| POST | `/api/evaluate/generate` | Generate checklist only |
| POST | `/api/evaluate/score` | Score with existing checklist items |

### Batch (`/api/batch`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/batch/upload` | Upload CSV/JSON file |
| POST | `/api/batch/upload-path` | Upload from server-side file path |
| POST | `/api/batch/{id}/start` | Start batch processing |
| GET | `/api/batch` | List all batches |
| GET | `/api/batch/{id}` | Get batch status + progress |
| GET | `/api/batch/{id}/results` | Get paginated results |
| POST | `/api/batch/{id}/cancel` | Cancel running batch |
| DELETE | `/api/batch/{id}` | Delete batch |
| GET | `/api/batch/{id}/export` | Export results (JSON/CSV) |

### Checklists (`/api/checklists`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/checklists` | List saved checklists |
| POST | `/api/checklists` | Save a new checklist |
| GET | `/api/checklists/{id}` | Get checklist detail |
| PUT | `/api/checklists/{id}` | Update checklist |
| DELETE | `/api/checklists/{id}` | Delete checklist |
| POST | `/api/checklists/import` | Import from JSON file |

### Prompt Templates (`/api/prompt-templates`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/prompt-templates` | List templates |
| POST | `/api/prompt-templates` | Create template |
| GET | `/api/prompt-templates/{id}` | Get template detail |
| PUT | `/api/prompt-templates/{id}` | Update template |
| DELETE | `/api/prompt-templates/{id}` | Delete template |
| POST | `/api/prompt-templates/seed-defaults` | Seed built-in default prompts |

### Pipelines (`/api/pipelines`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pipelines` | List pipelines |
| POST | `/api/pipelines` | Create pipeline |
| GET | `/api/pipelines/{id}` | Get pipeline config |
| PUT | `/api/pipelines/{id}` | Update pipeline |
| DELETE | `/api/pipelines/{id}` | Delete pipeline |

### Settings & Utilities
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/settings` | Get current settings (keys masked) |
| PUT | `/api/settings` | Update settings |
| POST | `/api/settings/test` | Test LLM provider connection |
| DELETE | `/api/settings/data` | Clear all stored data |
| GET | `/api/health` | Health check |
| GET | `/api/registry` | Available generators and scorers |
| GET | `/api/registry/preset-prompt/{name}` | Get preset generator prompt |
| GET | `/api/registry/scorer-prompt/{name}` | Get scorer prompt text |
| GET | `/api/registry/format/{name}` | Get output format template |
| GET | `/api/examples` | Pre-defined evaluation examples |

## Tests

### Backend tests (no API key needed)

```bash
uv run pytest ui/backend/tests/ -v
```

150+ unit tests covering all API endpoints, schemas, and services. Uses mocked LLM calls.

### Frontend build check

```bash
cd ui/frontend && npm run build
```

### Library tests (requires API key)

```bash
uv run pytest tests/ -v
```

**Note:** Run library and backend tests separately — they have conflicting `conftest.py` files.




