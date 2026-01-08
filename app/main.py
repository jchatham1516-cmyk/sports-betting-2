"""FastAPI entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import bets, runs
from app.db.session import init_db

app = FastAPI(title="Sports Betting API")

app.add_middleware(
    CORSMiddleware,
    # ✅ Add your Render domain so the app/other clients can call it cleanly
    # ✅ Keep localhost origins for local dev
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://sports-betting-2-ickm.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api")
app.include_router(bets.router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# ✅ NEW: Make the root URL "work" by sending people to Swagger docs
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}
