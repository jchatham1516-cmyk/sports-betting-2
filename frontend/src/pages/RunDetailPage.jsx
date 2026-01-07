import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  fetchPredictions,
  fetchRun,
  predictionDownloadUrl,
  trackedBetsDownloadUrl
} from "../api.js";

const RunDetailPage = () => {
  const { runId } = useParams();
  const [run, setRun] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        const runData = await fetchRun(runId);
        setRun(runData);
        const predictionData = await fetchPredictions(runId);
        setPredictions(predictionData);
      } catch (err) {
        setError(err.message);
      }
    };
    load();
  }, [runId]);

  const columns = useMemo(() => {
    if (predictions.length === 0) {
      return [];
    }
    return Object.keys(predictions[0]);
  }, [predictions]);

  if (error) {
    return <p style={{ color: "crimson" }}>{error}</p>;
  }

  return (
    <section>
      <h2>Run Detail</h2>
      {run ? (
        <div style={{ marginBottom: "1rem" }}>
          <p>
            <strong>Run ID:</strong> {run.id}
          </p>
          <p>
            <strong>Status:</strong> {run.status}
          </p>
          <p>
            <strong>Sport:</strong> {run.sport}
          </p>
          <p>
            <strong>Game Date:</strong> {run.game_date}
          </p>
        </div>
      ) : (
        <p>Loading run...</p>
      )}

      <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <a href={predictionDownloadUrl(runId)}>Download Predictions CSV</a>
        <a href={trackedBetsDownloadUrl(runId)}>Download Tracked Bets CSV</a>
      </div>

      {predictions.length === 0 ? (
        <p>No predictions available.</p>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table border="1" cellPadding="6" cellSpacing="0">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {predictions.map((row, index) => (
                <tr key={index}>
                  {columns.map((col) => (
                    <td key={col}>{String(row[col] ?? "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default RunDetailPage;
