import { useState } from "react";
import { fetchBets } from "../api.js";

const BetsPage = () => {
  const [sport, setSport] = useState("");
  const [date, setDate] = useState("");
  const [bets, setBets] = useState([]);
  const [error, setError] = useState("");

  const loadBets = async (event) => {
    event.preventDefault();
    setError("");
    try {
      const payload = await fetchBets({ date, sport: sport || undefined });
      setBets(payload);
    } catch (err) {
      setError(err.message);
      setBets([]);
    }
  };

  return (
    <section>
      <h2>Tracked Bets</h2>
      <form onSubmit={loadBets} style={{ display: "grid", gap: "0.75rem", maxWidth: 360 }}>
        <label>
          Date (YYYY-MM-DD)
          <input type="date" value={date} onChange={(event) => setDate(event.target.value)} required />
        </label>
        <label>
          Sport (optional)
          <select value={sport} onChange={(event) => setSport(event.target.value)}>
            <option value="">All</option>
            <option value="nba">NBA</option>
            <option value="nfl">NFL</option>
            <option value="nhl">NHL</option>
          </select>
        </label>
        <button type="submit">Load Bets</button>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
      </form>

      {bets.length === 0 ? (
        <p style={{ marginTop: "1rem" }}>No tracked bets for that date.</p>
      ) : (
        <div style={{ overflowX: "auto", marginTop: "1rem" }}>
          <table border="1" cellPadding="6" cellSpacing="0">
            <thead>
              <tr>
                <th>Sport</th>
                <th>Market</th>
                <th>Home</th>
                <th>Away</th>
                <th>Pick</th>
                <th>Price</th>
                <th>Units</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {bets.map((bet) => (
                <tr key={bet.id}>
                  <td>{bet.sport}</td>
                  <td>{bet.market}</td>
                  <td>{bet.home}</td>
                  <td>{bet.away}</td>
                  <td>{bet.pick}</td>
                  <td>{bet.price ?? ""}</td>
                  <td>{bet.units ?? ""}</td>
                  <td>{bet.result}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default BetsPage;
