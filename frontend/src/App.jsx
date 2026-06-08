import React, { useState } from "react";
import "./App.css";
import Dashboard from "./components/Dashboard";
import DatasetUpload from "./components/DatasetUpload";
import TrainingHub from "./components/TrainingHub";
import ModelRegistry from "./components/ModelRegistry";
import InferencePortal from "./components/InferencePortal";
import { LayoutDashboard, UploadCloud, PlayCircle, Award, Compass } from "lucide-react";

function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [datasetContextId, setDatasetContextId] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":
        return <Dashboard setActiveTab={setActiveTab} setJobContextId={setActiveJobId} />;
      case "upload":
        return <DatasetUpload setActiveTab={setActiveTab} setDatasetContextId={setDatasetContextId} />;
      case "train":
        return (
          <TrainingHub
            activeJobId={activeJobId}
            setActiveJobId={setActiveJobId}
            datasetContextId={datasetContextId}
          />
        );
      case "models":
        return <ModelRegistry />;
      case "inference":
        return <InferencePortal />;
      default:
        return <Dashboard setActiveTab={setActiveTab} setJobContextId={setActiveJobId} />;
    }
  };

  return (
    <div>
      {/* Header Navigation */}
      <nav className="navbar navbar-expand-lg navbar-dark navbar-custom">
        <div className="container">
          <a className="navbar-brand fw-extrabold text-white d-flex align-items-center gap-2" href="#" onClick={() => setActiveTab("dashboard")}>
            <span className="p-2 rounded-3" style={{ background: "linear-gradient(135deg, var(--color-indigo), var(--color-violet))" }}>
              <Compass size={20} className="text-white" />
            </span>
            <span className="glow-text fw-bold">YOLO Train & Detect</span>
          </a>
          
          <button className="navbar-toggler border-0" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span className="navbar-toggler-icon"></span>
          </button>
          
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
              <li className="nav-item">
                <a 
                  className={`nav-link nav-link-custom d-flex align-items-center gap-2 ${activeTab === "dashboard" ? "active" : ""}`} 
                  href="#" 
                  onClick={() => setActiveTab("dashboard")}
                >
                  <LayoutDashboard size={16} /> Dashboard
                </a>
              </li>
              <li className="nav-item">
                <a 
                  className={`nav-link nav-link-custom d-flex align-items-center gap-2 ${activeTab === "upload" ? "active" : ""}`} 
                  href="#" 
                  onClick={() => setActiveTab("upload")}
                >
                  <UploadCloud size={16} /> Upload Dataset
                </a>
              </li>
              <li className="nav-item">
                <a 
                  className={`nav-link nav-link-custom d-flex align-items-center gap-2 ${activeTab === "train" ? "active" : ""}`} 
                  href="#" 
                  onClick={() => setActiveTab("train")}
                >
                  <PlayCircle size={16} /> Training Hub
                </a>
              </li>
              <li className="nav-item">
                <a 
                  className={`nav-link nav-link-custom d-flex align-items-center gap-2 ${activeTab === "models" ? "active" : ""}`} 
                  href="#" 
                  onClick={() => setActiveTab("models")}
                >
                  <Award size={16} /> Model Registry
                </a>
              </li>
              <li className="nav-item">
                <a 
                  className={`nav-link nav-link-custom d-flex align-items-center gap-2 ${activeTab === "inference" ? "active" : ""}`} 
                  href="#" 
                  onClick={() => setActiveTab("inference")}
                >
                  <Compass size={16} /> Inference Portal
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>

      {/* Primary Tab View Container */}
      <main className="py-4">
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="py-4 mt-5 border-top border-secondary text-center small text-muted">
        <div className="container">
          <p className="mb-0">YOLO Custom Fine-Tuning & Container OCR Portal. Built using FastAPI & React.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
