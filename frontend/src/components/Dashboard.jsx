import React, { useEffect, useState } from "react";
import { getDashboardStats, API_BASE_URL } from "../services/api";
import { Cpu, Database, Award, Activity, Image as ImageIcon, CheckCircle, Clock } from "lucide-react";

export default function Dashboard({ setActiveTab, setJobContextId }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const data = await getDashboardStats();
      setStats(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to load dashboard metrics. Ensure the backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    // Poll stats every 10 seconds to keep active jobs updated
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStatusClass = (status) => {
    switch (status?.toUpperCase()) {
      case "COMPLETED": return "status-completed";
      case "TRAINING": return "status-training";
      case "PENDING": return "status-pending";
      case "FAILED": return "status-failed";
      default: return "status-pending";
    }
  };

  if (loading && !stats) {
    return (
      <div className="d-flex justify-content-center align-items-center py-5">
        <div className="spinner-border text-indigo" role="status">
          <span className="visually-hidden">Loading Dashboard...</span>
        </div>
      </div>
    );
  }

  const latestJob = stats?.latest_job;
  const latestInf = stats?.latest_inference;

  return (
    <div className="container py-4">
      {error && (
        <div className="alert alert-danger glass-panel border-danger d-flex align-items-center gap-2 mb-4" role="alert">
          <Activity size={20} className="text-danger" />
          <div>{error}</div>
        </div>
      )}

      {/* Metrics Row */}
      <div className="row g-4 mb-5">
        <div className="col-md-4">
          <div className="glass-panel p-4 d-flex align-items-center gap-3">
            <div className="p-3 rounded-3 bg-indigo-light" style={{ backgroundColor: "rgba(99, 102, 241, 0.1)", color: "#6366f1" }}>
              <Database size={28} />
            </div>
            <div>
              <p className="text-muted mb-1 fs-6 fw-medium">Datasets Uploaded</p>
              <h2 className="mb-0 fw-bold glow-text">{stats?.total_datasets || 0}</h2>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="glass-panel p-4 d-flex align-items-center gap-3">
            <div className="p-3 rounded-3 bg-teal-light" style={{ backgroundColor: "rgba(20, 184, 166, 0.1)", color: "#14b8a6" }}>
              <Cpu size={28} />
            </div>
            <div>
              <p className="text-muted mb-1 fs-6 fw-medium">Trained Models</p>
              <h2 className="mb-0 fw-bold glow-teal">{stats?.total_models || 0}</h2>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="glass-panel p-4 d-flex align-items-center gap-3">
            <div className="p-3 rounded-3 bg-violet-light" style={{ backgroundColor: "rgba(139, 92, 246, 0.1)", color: "#8b5cf6" }}>
              <Award size={28} />
            </div>
            <div>
              <p className="text-muted mb-1 fs-6 fw-medium">Active Model Status</p>
              <h5 className="mb-0 fw-bold text-white text-truncate" style={{ maxWidth: "200px" }}>
                {stats?.total_models > 0 ? "Ready for Inference" : "No Models Loaded"}
              </h5>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-4">
        {/* Left Side - Latest Job Details */}
        <div className="col-lg-6">
          <div className="glass-panel h-100">
            <div className="glass-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
                <Clock size={18} className="text-indigo" />
                Latest Training Job
              </h5>
              {latestJob && (
                <span className={`status-badge ${getStatusClass(latestJob.status)}`}>
                  {latestJob.status}
                </span>
              )}
            </div>
            <div className="p-4">
              {latestJob ? (
                <div>
                  <div className="mb-3 d-flex justify-content-between">
                    <div>
                      <span className="text-muted d-block small">Job Reference</span>
                      <span className="fw-semibold text-white">Job #{latestJob.id}</span>
                    </div>
                    <div>
                      <span className="text-muted d-block text-end small">Base Model</span>
                      <span className="fw-semibold text-white">{latestJob.base_model}</span>
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="d-flex justify-content-between mb-1 small">
                      <span className="text-muted">Progress ({latestJob.progress_percent}%)</span>
                      <span className="text-white">Epoch {latestJob.current_epoch}/{latestJob.epochs}</span>
                    </div>
                    <div className="progress-custom">
                      <div 
                        className="progress-bar-custom h-100" 
                        style={{ width: `${latestJob.progress_percent}%` }}
                      ></div>
                    </div>
                  </div>

                  {latestJob.status === "TRAINING" && (
                    <div className="alert alert-info bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-center shimmer-bg text-white">
                      Training in progress. Status refreshes automatically...
                    </div>
                  )}

                  {latestJob.error_message && (
                    <div className="alert alert-danger bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-danger small">
                      <strong>Error:</strong> {latestJob.error_message}
                    </div>
                  )}

                  <div className="d-flex gap-2">
                    <button 
                      onClick={() => {
                        setJobContextId(latestJob.id);
                        setActiveTab("train");
                      }} 
                      className="btn btn-indigo flex-grow-1"
                    >
                      View Details
                    </button>
                    <button 
                      onClick={() => setActiveTab("train")} 
                      className="btn btn-glass"
                    >
                      Config Runs
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-5">
                  <Cpu size={48} className="text-muted mb-3 opacity-20" />
                  <p className="text-muted">No training runs registered yet.</p>
                  <button onClick={() => setActiveTab("upload")} className="btn btn-indigo mt-2">
                    Upload Dataset to Start
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Side - Latest Inference Result */}
        <div className="col-lg-6">
          <div className="glass-panel h-100">
            <div className="glass-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
                <ImageIcon size={18} className="text-teal" />
                Latest Detection Results
              </h5>
              {latestInf && (
                <span className="status-badge status-completed">
                  <CheckCircle size={12} /> Detected
                </span>
              )}
            </div>
            <div className="p-4">
              {latestInf ? (
                <div>
                  <div className="row g-3 align-items-center mb-4">
                    <div className="col-md-5">
                      <div className="position-relative overflow-hidden rounded-3 border border-secondary" style={{ maxHeight: "160px" }}>
                        <img 
                          src={`${API_BASE_URL}${latestInf.annotated_url}`} 
                          alt="Annotated Inference" 
                          className="img-fluid w-100" 
                          style={{ objectFit: "cover" }}
                        />
                      </div>
                    </div>
                    <div className="col-md-7">
                      <span className="text-muted small d-block">Container Number</span>
                      <h4 className="fw-bold text-teal glow-teal mb-2 font-monospace">
                        {latestInf.detections[0]?.text || latestInf.container_number || "No text detected"}
                      </h4>
                      <div className="d-flex justify-content-between border-top border-secondary pt-2">
                        <div>
                          <span className="text-muted small d-block">Detections</span>
                          <span className="fw-semibold text-white">{latestInf.detections.length} objects</span>
                        </div>
                        <div>
                          <span className="text-muted small d-block">Confidence</span>
                          <span className="fw-semibold text-white">
                            {latestInf.detections[0] ? `${(latestInf.detections[0].confidence * 100).toFixed(1)}%` : "N/A"}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <button onClick={() => setActiveTab("inference")} className="btn btn-indigo w-100">
                    Open Inference Hub
                  </button>
                </div>
              ) : (
                <div className="text-center py-5">
                  <ImageIcon size={48} className="text-muted mb-3 opacity-20" />
                  <p className="text-muted">No test images run yet.</p>
                  <button onClick={() => setActiveTab("inference")} className="btn btn-indigo mt-2">
                    Test Model Bounding Boxes
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
