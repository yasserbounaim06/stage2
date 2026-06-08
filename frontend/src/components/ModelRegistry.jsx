import React, { useEffect, useState } from "react";
import { getModels, activateModel } from "../services/api";
import { Award, CheckCircle, RefreshCw, Calendar, Tag, ShieldAlert } from "lucide-react";

export default function ModelRegistry() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actLoadingId, setActLoadingId] = useState(null);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const data = await getModels();
      setModels(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to retrieve the model registry list.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleActivate = async (modelId) => {
    try {
      setActLoadingId(modelId);
      await activateModel(modelId);
      // Re-fetch list to update active badges
      const data = await getModels();
      setModels(data);
    } catch (err) {
      console.error(err);
      alert("Failed to activate model version.");
    } finally {
      setActLoadingId(null);
    }
  };

  const getMetricValue = (metrics, key) => {
    if (!metrics) return "N/A";
    const parsed = typeof metrics === "string" ? JSON.parse(metrics) : metrics;
    const latest = parsed.latest || {};
    // Try clean key first, then fallback
    return latest[key]?.toFixed(4) || latest[key.replace("_", "/")]?.toFixed(4) || "N/A";
  };

  if (loading && models.length === 0) {
    return (
      <div className="d-flex justify-content-center align-items-center py-5">
        <div className="spinner-border text-indigo" role="status">
          <span className="visually-hidden">Loading model version history...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="container py-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h4 className="fw-bold text-white mb-1">Model Version Registry</h4>
          <p className="text-muted small mb-0">Select and activate trained weights for inference pipelines.</p>
        </div>
        <button onClick={fetchModels} className="btn btn-glass d-flex align-items-center gap-2">
          <RefreshCw size={14} /> Refresh List
        </button>
      </div>

      {error && (
        <div className="alert alert-danger bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-danger small">
          {error}
        </div>
      )}

      {models.length === 0 ? (
        <div className="glass-panel p-5 text-center">
          <ShieldAlert size={48} className="text-muted mb-3 opacity-20" />
          <h5 className="fw-bold text-white mb-2">No Models Registered</h5>
          <p className="text-muted mb-0">Complete a YOLO training run to register custom weights in this list.</p>
        </div>
      ) : (
        <div className="row g-4">
          {models.map((model) => (
            <div className="col-md-6 col-lg-4" key={model.id}>
              <div className={`glass-panel h-100 position-relative ${model.is_active ? "glow-border-active" : ""}`}>
                <div className="glass-header d-flex justify-content-between align-items-start">
                  <div className="text-truncate" style={{ maxWidth: "70%" }}>
                    <span className="text-muted small d-block mb-1">Model Version</span>
                    <h5 className="fw-bold text-white mb-0 text-truncate" title={model.version_name}>
                      {model.version_name}
                    </h5>
                  </div>
                  {model.is_active ? (
                    <span className="status-badge status-completed">
                      <CheckCircle size={12} /> Active
                    </span>
                  ) : (
                    <span className="status-badge status-pending text-muted border-secondary bg-transparent">
                      Inactive
                    </span>
                  )}
                </div>

                <div className="p-4">
                  {/* Info table */}
                  <div className="mb-4 small">
                    <div className="d-flex justify-content-between border-bottom border-secondary py-1 text-muted">
                      <span>Created</span>
                      <span className="text-white d-flex align-items-center gap-1">
                        <Calendar size={12} /> {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="d-flex justify-content-between border-bottom border-secondary py-1 text-muted">
                      <span>Final Loss</span>
                      <span className="text-white fw-medium">
                        {getMetricValue(model.metrics, "loss")}
                      </span>
                    </div>
                    <div className="d-flex justify-content-between border-bottom border-secondary py-1 text-muted">
                      <span>mAP50</span>
                      <span className="text-white fw-medium">
                        {getMetricValue(model.metrics, "metrics_mAP50B")}
                      </span>
                    </div>
                    <div className="d-flex justify-content-between py-1 text-muted">
                      <span>Precision</span>
                      <span className="text-white fw-medium">
                        {getMetricValue(model.metrics, "metrics_precisionB")}
                      </span>
                    </div>
                  </div>

                  {model.is_active ? (
                    <button className="btn btn-glass w-100" disabled>
                      Selected for Inference
                    </button>
                  ) : (
                    <button
                      onClick={() => handleActivate(model.id)}
                      className="btn btn-indigo w-100 d-flex align-items-center justify-content-center gap-2"
                      disabled={actLoadingId !== null}
                    >
                      {actLoadingId === model.id ? (
                        <>
                          <RefreshCw size={14} className="spin-animation" /> Activating...
                        </>
                      ) : (
                        <>
                          <Award size={14} /> Set as Active Model
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
