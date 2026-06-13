import React, { useState, useEffect } from "react";
import { startTraining, getTrainingStatus, getDashboardStats } from "../services/api";
import { Cpu, Play, Terminal, CheckCircle, AlertCircle, RefreshCw, BarChart2 } from "lucide-react";

export default function TrainingHub({ activeJobId, setActiveJobId, datasetContextId }) {
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [imgsz, setImgsz] = useState(640);
  const [baseModel, setBaseModel] = useState("yolov8n.pt");
  
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasetContextId || "");
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Sync selectedDatasetId with prop context if it updates
  useEffect(() => {
    if (datasetContextId) {
      setSelectedDatasetId(datasetContextId);
    }
  }, [datasetContextId]);

  // Poll job status if a job is running
  const fetchJobStatus = async () => {
    if (!activeJobId) return;
    try {
      const data = await getTrainingStatus(activeJobId);
      setJob(data);
      if (data.status === "COMPLETED" || data.status === "FAILED") {
        // Stop polling active job id once completed
        setActiveJobId(null);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to fetch training progress.");
    }
  };

  useEffect(() => {
    if (activeJobId) {
      fetchJobStatus();
      const interval = setInterval(fetchJobStatus, 3000);
      return () => clearInterval(interval);
    }
  }, [activeJobId]);

  // If there's an active job but we just opened the page, fetch it once
  useEffect(() => {
    if (activeJobId) {
      fetchJobStatus();
    }
  }, [activeJobId]);

  const handleStart = async (e) => {
    e.preventDefault();
    const dsId = Number(selectedDatasetId || datasetContextId);
    if (!dsId) {
      setError("Please select a validated dataset.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await startTraining(dsId, epochs, batchSize, imgsz, baseModel);
      setActiveJobId(data.id);
      setJob(data);
    } catch (err) {
      setError(err.message || "Failed to start training.");
    } finally {
      setLoading(false);
    }
  };

  const getStatusClass = (status) => {
    switch (status?.toUpperCase()) {
      case "COMPLETED": return "status-completed";
      case "TRAINING": return "status-training";
      case "PENDING": return "status-pending";
      case "FAILED": return "status-failed";
      default: return "status-pending";
    }
  };

  // Helper to render metrics history logs/graphs
  const renderMetricsLogs = () => {
    const jobMetrics = job?.metrics;
    let history = [];
    if (jobMetrics) {
      try {
        const parsed = typeof jobMetrics === "string" ? JSON.parse(jobMetrics) : jobMetrics;
        history = parsed.history || [];
      } catch (e) {
        console.error(e);
      }
    }

    if (!history || history.length === 0) {
      return <div className="text-muted small">Awaiting first epoch metrics...</div>;
    }

    return (
      <div className="terminal-block">
        {history.map((h, i) => (
          <div key={i} className="mb-1">
            <span className="text-teal">Epoch {h.epoch}:</span>{" "}
            Loss = <span className="text-white">{h.loss?.toFixed(4) || "N/A"}</span> |{" "}
            mAP50 = <span className="text-violet">{(h.metrics_mAP50B || h.map50 || 0).toFixed(4)}</span> |{" "}
            Precision = <span className="text-amber">{(h.metrics_precisionB || h.precision || 0).toFixed(4)}</span>
          </div>
        ))}
      </div>
    );
  };

  // Extract latest metrics
  const getLatestMetrics = () => {
    if (!job?.metrics) return null;
    try {
      const parsed = typeof job.metrics === "string" ? JSON.parse(job.metrics) : job.metrics;
      return parsed.latest || null;
    } catch (e) {
      return null;
    }
  };

  const latestMetrics = getLatestMetrics();

  return (
    <div className="container py-4">
      <div className="row g-4">
        {/* Left Side: Setup Training */}
        <div className="col-lg-5">
          <div className="glass-panel h-100">
            <div className="glass-header">
              <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
                <Cpu size={18} className="text-indigo" />
                Configure Fine-Tuning
              </h5>
            </div>

            <div className="p-4">
              {error && (
                <div className="alert alert-danger bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-danger small">
                  {error}
                </div>
              )}

              <form onSubmit={handleStart}>
                <div className="mb-3">
                  <label className="form-label text-muted small fw-semibold">Dataset ID</label>
                  {datasetContextId ? (
                    <div className="form-control form-control-custom text-white" style={{ background: "rgba(99, 102, 241, 0.05)" }}>
                      Dataset #{datasetContextId} (Selected Upload)
                    </div>
                  ) : (
                    <input
                      type="number"
                      placeholder="Enter Dataset ID"
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                      className="form-control form-control-custom"
                      required
                    />
                  )}
                  <span className="text-muted small mt-1 d-block">Ensure this dataset ID has passed validation checks.</span>
                </div>

                <div className="row g-3 mb-3">
                  <div className="col-6">
                    <label className="form-label text-muted small fw-semibold">Epochs</label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                      className="form-control form-control-custom"
                    />
                  </div>
                  <div className="col-6">
                    <label className="form-label text-muted small fw-semibold">Batch Size</label>
                    <input
                      type="number"
                      min="1"
                      max="64"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      className="form-control form-control-custom"
                    />
                  </div>
                </div>

                <div className="row g-3 mb-4">
                  <div className="col-6">
                    <label className="form-label text-muted small fw-semibold">Image Size (imgsz)</label>
                    <select
                      value={imgsz}
                      onChange={(e) => setImgsz(Number(e.target.value))}
                      className="form-select form-control-custom"
                    >
                      <option value="320">320</option>
                      <option value="416">416</option>
                      <option value="640">640</option>
                    </select>
                  </div>
                  <div className="col-6">
                    <label className="form-label text-muted small fw-semibold">Base Model</label>
                    <select
                      value={baseModel}
                      onChange={(e) => setBaseModel(e.target.value)}
                      className="form-select form-control-custom"
                    >
                      <option value="yolov8n.pt">yolov8n.pt (Nano)</option>
                      <option value="yolov8s.pt">yolov8s.pt (Small)</option>
                    </select>
                  </div>
                </div>

                <button
                  type="submit"
                  className="btn btn-indigo w-100 d-flex align-items-center justify-content-center gap-2"
                  disabled={loading || activeJobId !== null}
                >
                  {loading ? (
                    <>
                      <RefreshCw size={16} className="spin-animation" /> Launching Run...
                    </>
                  ) : (
                    <>
                      <Play size={16} fill="white" /> Start YOLO Training
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Right Side: Training Status Monitor */}
        <div className="col-lg-7">
          <div className="glass-panel h-100">
            <div className="glass-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
                <Terminal size={18} className="text-teal" />
                Training Run Monitor
              </h5>
              {job && (
                <span className={`status-badge ${getStatusClass(job.status)}`}>
                  {job.status}
                </span>
              )}
            </div>

            <div className="p-4">
              {job ? (
                <div>
                  <div className="mb-4">
                    <div className="d-flex justify-content-between mb-2">
                      <div>
                        <span className="text-muted d-block small">Training Run Reference</span>
                        <span className="fw-bold text-white">Job #{job.id}</span>
                      </div>
                      <div className="text-end">
                        <span className="text-muted d-block small">Epoch Progress</span>
                        <span className="fw-bold text-white">
                          {job.current_epoch} / {job.epochs}
                        </span>
                      </div>
                    </div>
                    
                    <div className="progress-custom mb-2">
                      <div 
                        className="progress-bar-custom h-100" 
                        style={{ width: `${job.progress_percent}%` }}
                      ></div>
                    </div>
                    <span className="text-muted small">Completion rate: {job.progress_percent}%</span>
                  </div>

                  {/* Real-time stats panels */}
                  {latestMetrics && (
                    <div className="row g-3 mb-4">
                      <div className="col-4">
                        <div className="p-3 bg-dark-opacity border border-secondary rounded-3 text-center">
                          <span className="text-muted small d-block mb-1">Loss</span>
                          <span className="fw-bold text-teal">{latestMetrics.loss?.toFixed(4) || "N/A"}</span>
                        </div>
                      </div>
                      <div className="col-4">
                        <div className="p-3 bg-dark-opacity border border-secondary rounded-3 text-center">
                          <span className="text-muted small d-block mb-1">mAP50</span>
                          <span className="fw-bold text-indigo">
                            {(latestMetrics.metrics_mAP50B || latestMetrics.map50 || 0).toFixed(3)}
                          </span>
                        </div>
                      </div>
                      <div className="col-4">
                        <div className="p-3 bg-dark-opacity border border-secondary rounded-3 text-center">
                          <span className="text-muted small d-block mb-1">Precision</span>
                          <span className="fw-bold text-violet">
                            {(latestMetrics.metrics_precisionB || latestMetrics.precision || 0).toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Terminal block logs */}
                  <h6 className="text-muted small fw-bold mb-2">Metrics Log history</h6>
                  {renderMetricsLogs()}

                  {job.status === "COMPLETED" && (
                    <div className="alert alert-success glass-panel border-success mt-4 p-3 d-flex align-items-center gap-2" role="alert">
                      <CheckCircle size={20} className="text-teal" />
                      <div>
                        <strong>Fine-tuning complete!</strong> Model weights registered as active. You can now use this model for inferences.
                      </div>
                    </div>
                  )}

                  {job.status === "FAILED" && (
                    <div className="alert alert-danger glass-panel border-danger mt-4 p-3 d-flex align-items-center gap-2" role="alert">
                      <AlertCircle size={20} className="text-rose" />
                      <div>
                        <strong>Training failed:</strong> {job.error_message}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-5">
                  <BarChart2 size={48} className="text-muted mb-3 opacity-20" />
                  <p className="text-muted">Awaiting job configuration and launch...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
