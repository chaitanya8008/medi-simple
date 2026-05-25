# Text-to-SQL — Natural Language Database Query Engine

An LLM-powered system that converts plain-English questions into validated, executed SQL queries with natural language responses. Supports multiple database dialects with built-in safety, retry logic, and ambiguity detection.

Ask `"Which customers placed orders worth more than $500 last month?"` → get the SQL, the results, and a human-readable answer.

**Stack:** Python · FastAPI · Groq (Llama 3.3) · PostgreSQL · SQLAlchemy (async) · Streamlit

---

## Architecture

```
User Question
    │
    ▼
┌─────────────────────┐
│  Ambiguity Detector  │  LLM checks if the question is too vague
│                     │  "show top products" → asks: "top by what?"
│                     │  Clear questions pass through
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Schema Loader       │  SQLAlchemy inspect → table names, column types
│                     │  Loads sample rows for few-shot context
│                     │  Smart keyword matching: only loads relevant tables
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Prompt Builder      │  Dialect-aware prompt (PostgreSQL/MySQL/SQLite/MSSQL)
│                     │  Few-shot examples + schema context + safety rules
│                     │  Supports follow-up queries via last_sql context
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  SQL Generator       │  Groq LLM with system-level safety prompt
│                     │  Read-only enforcement: blocks INSERT/UPDATE/DELETE/DROP
│                     │  Returns UNABLE_TO_ANSWER for out-of-scope questions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Validation Layer    │  Checks generated table names against actual schema
│                     │  Auto-enforces LIMIT (dialect-aware: LIMIT vs TOP)
│                     │  Catches hallucinated table/column names before execution
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Retry Engine        │  If query fails → sends error + original question
│                     │  back to LLM for self-correction
│                     │  Up to N retries with full error context
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Response Generator  │  Converts raw SQL results into natural language
│                     │  "There are 12 customers who placed orders over $500"
└─────────────────────┘
```

## Key Design Decisions

**Why validate SQL against the schema before executing?**

LLMs hallucinate table and column names. The `validate_sql_against_schema` function uses SQLAlchemy's `inspect` to extract real table names from the database, then checks every `FROM` and `JOIN` target in the generated SQL. Hallucinated tables get caught before they hit the database, saving a round-trip and producing a better error message for the retry prompt.

**Why dialect-aware prompting?**

SQL isn't universal — `LIMIT 50` works in PostgreSQL but SQL Server needs `SELECT TOP 50`. The prompt builder injects dialect-specific rules (type casting syntax, pagination, identifier quoting) so the LLM generates syntactically correct SQL for whichever database is connected. Switching databases requires changing one config value, not rewriting prompts.

**Why ambiguity detection before query generation?**

Vague questions like *"show me the best products"* produce arbitrary SQL (best by revenue? by rating? by quantity sold?). The `detect_ambiguity` function asks the LLM to evaluate the question first. If it's ambiguous, the system returns a clarifying question instead of guessing — which saves LLM tokens and avoids misleading results.

**Why retry with error context instead of just failing?**

First-attempt SQL often has minor syntax errors or wrong joins. Instead of returning an error to the user, the retry engine sends the failed SQL + the exact database error message + the original question back to the LLM, which is usually enough context for self-correction. This handles ~80% of first-attempt failures transparently.

**Safety model:**
- System prompt enforces read-only mode at the LLM level
- Regex-based write query detection as a second layer
- Automatic `LIMIT` enforcement prevents unbounded result sets
- Schema validation catches hallucinated table names

## API

```
POST /api/query    — Send a question, get SQL + results + natural language answer
GET  /api/health   — Health check
```

## Setup

```bash
# Clone and install
git clone https://github.com/chaitanya8008/text-to-SQL.git
cd text-to-SQL
pip install -r requirements.txt

# Configure environment
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=your_database_connection_string

# Run backend
uvicorn backend.main:app --reload

# Run frontend (separate terminal)
streamlit run frontend/app.py
```

## Project Structure

```
text-to-sql/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoint with full query pipeline
│   ├── core/
│   │   └── config.py          # DB config, few-shot examples, settings
│   ├── db/
│   │   ├── database.py        # Async query execution, retry logic, validation
│   │   └── schema_loader.py   # SQLAlchemy inspect, smart table filtering
│   ├── llm/
│   │   ├── groq_client.py     # SQL generation, NL response, ambiguity detection
│   │   └── prompt_builder.py  # Dialect-aware prompt construction
│   ├── utils/
│   │   └── debug_logger.py    # Query debug logging
│   └── main.py
├── frontend/
│   └── app.py                 # Streamlit chat interface
├── sample_db/
│   ├── create_db.py
│   └── add_complex_table.py
├── requirements.txt
└── README.md
```
