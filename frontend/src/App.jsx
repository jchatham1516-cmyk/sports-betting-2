import { Link, Route, Routes } from "react-router-dom";
import RunPage from "./pages/RunPage.jsx";
import RunDetailPage from "./pages/RunDetailPage.jsx";
import BetsPage from "./pages/BetsPage.jsx";

const App = () => {
  return (
    <div style={{ fontFamily: "Arial, sans-serif", padding: "1.5rem" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <h1>Sports Betting App</h1>
        <nav style={{ display: "flex", gap: "1rem" }}>
          <Link to="/">Run</Link>
          <Link to="/bets">Bets</Link>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<RunPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/bets" element={<BetsPage />} />
      </Routes>
    </div>
  );
};

export default App;
