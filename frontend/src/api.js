const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const runModel = async ({ sport, game_date }) => {
  const response = await fetch(`${API_BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sport, game_date })
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error?.detail?.message || "Failed to run model");
  }
  return response.json();
};

export const fetchRun = async (runId) => {
  const response = await fetch(`${API_BASE}/api/runs/${runId}`);
  if (!response.ok) {
    throw new Error("Run not found");
  }
  return response.json();
};

export const fetchPredictions = async (runId) => {
  const response = await fetch(`${API_BASE}/api/runs/${runId}/predictions`);
  if (!response.ok) {
    throw new Error("Failed to load predictions");
  }
  return response.json();
};

export const fetchBets = async ({ date, sport }) => {
  const url = new URL(`${API_BASE}/api/bets`);
  url.searchParams.set("date", date);
  if (sport) {
    url.searchParams.set("sport", sport);
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Failed to load tracked bets");
  }
  return response.json();
};

export const predictionDownloadUrl = (runId) =>
  `${API_BASE}/api/runs/${runId}/download/predictions.csv`;

export const trackedBetsDownloadUrl = (runId) =>
  `${API_BASE}/api/runs/${runId}/download/tracked_bets.csv`;
