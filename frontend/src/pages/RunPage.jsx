import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { runModel } from "../api.js";

const RunPage = () => {
  const [sport, setSport] = useState("nba");
  const [gameDate, setGameDate] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const payload = await runModel({
        sport,
        game_date: gameDate
      });
      navigate(`/runs/${payload.run_id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <h2>Run Model</h2>
      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "0.75rem", maxWidth: 360 }}>
        <label>
          Sport
          <select value={sport} onChange={(event) => setSport(event.target.value)}>
            <option value="nba">NBA</option>
            <option value="nfl">NFL</option>
            <option value="nhl">NHL</option>
          </select>
        </label>
        <label>
          Game Date (YYYY-MM-DD)
          <input
            type="date"
            value={gameDate}
            onChange={(event) => setGameDate(event.target.value)}
            required
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
      </form>
    </section>
  );
};

export default RunPage;
