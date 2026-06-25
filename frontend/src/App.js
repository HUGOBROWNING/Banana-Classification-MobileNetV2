import { useEffect, useState } from "react";
import "@/App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

function App() {
  const [status, setStatus] = useState("checking");
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios
      .get(`${API}/`)
      .then((response) => {
        setStatus("connected");
        setMessage(response.data.message);
      })
      .catch(() => {
        setStatus("disconnected");
        setMessage(`Could not reach backend at ${BACKEND_URL}`);
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🍌 Banana Ripeness Classifier</h1>
        <p
          style={{
            color:
              status === "connected"
                ? "#4C9A2A"
                : status === "disconnected"
                  ? "#C07A1E"
                  : "#888",
          }}
        >
          Backend:{" "}
          {status === "checking"
            ? "Checking connection…"
            : status === "connected"
              ? `Connected — ${message}`
              : message}
        </p>
        <p style={{ marginTop: "1.5rem", fontSize: "0.95rem" }}>
          Open the classifier app at{" "}
          <a href="http://localhost:8501" style={{ color: "#E29400" }}>
            http://localhost:8501
          </a>
        </p>
      </header>
    </div>
  );
}

export default App;
