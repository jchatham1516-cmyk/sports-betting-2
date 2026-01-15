# Sports Betting App

## Backend (FastAPI)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Environment Variables

- `DATABASE_URL`: SQLAlchemy-compatible database URL. Defaults to `sqlite:///app/db/app.db`.
- `RESULTS_DIR`: Override the results directory (defaults to `results`).
- `ODDS_DIR`: Override the odds directory (defaults to `odds`).
- `ODDS_API_KEY`: Odds API key used by the model (if required by your odds provider).
- `NHL_HOME_ADV`: Elo home-ice advantage in points (default `45.0`).
- `NHL_ELO_K`: Elo K-factor for NHL updates (default `18.0`).
- `NHL_STRICT_SANITY`: Set to `1` to raise if NHL probabilities look constant.

## NHL Model Notes

### Train/refresh NHL Elo

Run the NHL Elo updater directly to refresh ratings from recent completed games:

```bash
python -c "from sports.nhl.model import update_elo_from_recent_scores; update_elo_from_recent_scores(120)"
```

### Goalie cache location

Starting goalie data is cached under:

```
results/cache/nhl_goalies_YYYY-MM-DD.json
```

## Frontend (Vite React)

### Setup

```bash
cd frontend
npm install
```

### Run

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173` and expects the API at `http://localhost:8000`. You can override the API URL via `VITE_API_URL`.

## API Endpoints

- `POST /api/runs` – Run the model pipeline and persist predictions/tracked bets.
- `GET /api/runs/{run_id}` – Fetch run metadata.
- `GET /api/runs/{run_id}/predictions` – Fetch prediction rows.
- `GET /api/runs/{run_id}/download/predictions.csv` – Download predictions CSV.
- `GET /api/runs/{run_id}/download/tracked_bets.csv` – Download tracked bets CSV (if available).
- `GET /api/bets?date=YYYY-MM-DD&sport=nba` – List tracked bets for a date.
- `POST /api/bets/settle?date=YYYY-MM-DD&sport=nba` – Placeholder settlement endpoint.

## Deploy on Render (Docker)

1. Create a new Render Web Service.
2. Choose Docker as the runtime.
3. Add environment variables (ex: `DATABASE_URL`, `ODDS_API_KEY`).
4. Configure the start command:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 10000
   ```
5. Point Render to the repo root so it can access the Dockerfile you provide.

To deploy the frontend, create a Render Static Site that runs:

```bash
cd frontend
npm install
npm run build
```

and uses `frontend/dist` as the publish directory.
