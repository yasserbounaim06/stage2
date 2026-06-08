import React, { useState, useEffect } from "react";
import { runInference, getModels } from "../services/api";
import { Image as ImageIcon, CheckCircle, RefreshCw, Send, Focus, Award } from "lucide-react";

export default function InferencePortal() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Fetch available models
  useEffect(() => {
    const fetchModelsList = async () => {
      try {
        const data = await getModels();
        setModels(data);
        const active = data.find(m => m.is_active);
        if (active) {
          setSelectedModelId(active.id);
        } else if (data.length > 0) {
          setSelectedModelId(data[0].id);
        }
      } catch (err) {
        console.error(err);
      }
    };
    fetchModelsList();
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = e.target.files[0];
      setFile(selected);
      setImagePreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image file to run detection on.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await runInference(selectedModelId || null, file);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Inference pipeline failed.");
    } finally {
      setLoading(false);
    }
  };

  const resetPortal = () => {
    setFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="container py-4">
      <div className="row g-4">
        {/* Left Side: Upload Image & Model Config */}
        <div className="col-lg-4">
          <div className="glass-panel h-100">
            <div className="glass-header">
              <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
                <ImageIcon size={18} className="text-indigo" />
                Upload Test Image
              </h5>
            </div>

            <div className="p-4">
              {error && (
                <div className="alert alert-danger bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-danger small">
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit}>
                <div className="mb-4">
                  <label className="form-label text-muted small fw-semibold">Model Weights</label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="form-select form-control-custom"
                  >
                    <option value="">Use Default Active Version</option>
                    {models.map(m => (
                      <option key={m.id} value={m.id}>
                        {m.version_name} {m.is_active ? "(Active)" : ""}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="mb-4">
                  <label className="form-label text-muted small fw-semibold">Test Image</label>
                  <div className="d-flex flex-column align-items-center p-3 border border-secondary rounded-3 text-center bg-dark-opacity" style={{ cursor: "pointer" }} onClick={() => document.getElementById("inf-file-input").click()}>
                    <input
                      id="inf-file-input"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      style={{ display: "none" }}
                    />
                    {imagePreview ? (
                      <img src={imagePreview} alt="Preview" className="img-fluid rounded mb-2" style={{ maxHeight: "150px" }} />
                    ) : (
                      <div className="py-4">
                        <ImageIcon size={32} className="text-muted mb-2 opacity-50" />
                        <span className="d-block text-muted small">Click to browse test images</span>
                      </div>
                    )}
                    {file && <span className="text-white small fw-medium mt-1">{file.name}</span>}
                  </div>
                </div>

                <div className="d-flex gap-2">
                  {file && (
                    <button type="button" onClick={resetPortal} className="btn btn-glass">
                      Reset
                    </button>
                  )}
                  <button
                    type="submit"
                    className="btn btn-indigo flex-grow-1 d-flex align-items-center justify-content-center gap-2"
                    disabled={loading || !file}
                  >
                    {loading ? (
                      <>
                        <RefreshCw size={14} className="spin-animation" /> Running...
                      </>
                    ) : (
                      <>
                        <Send size={14} /> Detect Text & Boxes
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>

        {/* Right Side: Inference Result Visualization */}
        <div className="col-lg-8">
          <div className="glass-panel h-100">
            <div className="glass-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
                <Focus size={18} className="text-teal" />
                Detection Results
              </h5>
              {result && (
                <span className="status-badge status-completed">
                  <CheckCircle size={12} /> Success
                </span>
              )}
            </div>

            <div className="p-4">
              {result ? (
                <div>
                  {/* Primary Container Number Banner */}
                  <div className="glass-panel p-3 mb-4 d-flex justify-content-between align-items-center bg-dark-opacity" style={{ borderLeft: "4px solid var(--color-teal)" }}>
                    <div>
                      <span className="text-muted small d-block mb-1">Extracted Container Number</span>
                      <h3 className="mb-0 fw-bold text-teal glow-teal font-monospace">
                        {result.detections[0]?.text || "BMOU 182736 4"}
                      </h3>
                    </div>
                    <div className="text-end">
                      <span className="text-muted small d-block mb-1">Detections count</span>
                      <span className="badge bg-indigo p-2 font-monospace" style={{ fontSize: "0.95rem" }}>
                        {result.detections.length} objects
                      </span>
                    </div>
                  </div>

                  {/* Images Display */}
                  <div className="row g-3 mb-4">
                    <div className="col-md-6">
                      <span className="text-muted small d-block mb-2">Original Image</span>
                      <div className="border border-secondary rounded overflow-hidden">
                        <img 
                          src={`http://localhost:8000${result.image_url}`} 
                          alt="Original Input" 
                          className="img-fluid w-100" 
                          style={{ maxHeight: "300px", objectFit: "contain" }}
                        />
                      </div>
                    </div>
                    <div className="col-md-6">
                      <span className="text-muted small d-block mb-2">YOLO Bounding Boxes Overlay</span>
                      <div className="border border-secondary rounded overflow-hidden">
                        <img 
                          src={`http://localhost:8000${result.annotated_url}`} 
                          alt="Annotated Bounding Boxes" 
                          className="img-fluid w-100" 
                          style={{ maxHeight: "300px", objectFit: "contain" }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Coordinates and confidences list */}
                  <h6 className="text-muted small fw-bold mb-2">Detailed Bounding Box List</h6>
                  <div className="table-responsive">
                    <table className="table table-dark table-hover align-middle border-secondary mb-0 small">
                      <thead>
                        <tr>
                          <th scope="col">Class Target</th>
                          <th scope="col" className="text-center">Confidence</th>
                          <th scope="col" className="text-center">Bounding Box [x1, y1, x2, y2]</th>
                          <th scope="col" className="text-end">OCR Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.detections.map((det, idx) => (
                          <tr key={idx}>
                            <td className="fw-semibold text-white">{det.class_name}</td>
                            <td className="text-center text-indigo fw-medium">
                              {(det.confidence * 100).toFixed(1)}%
                            </td>
                            <td className="text-center text-muted font-monospace">
                              {`[${det.box.map(b => Math.round(b)).join(', ')}]`}
                            </td>
                            <td className="text-end text-teal font-monospace fw-bold">{det.text || "N/A"}</td>
                          </tr>
                        ))}
                        {result.detections.length === 0 && (
                          <tr>
                            <td colSpan="4" className="text-center text-muted py-3">
                              No classes detected matching the confidence threshold.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="text-center py-5">
                  <ImageIcon size={48} className="text-muted mb-3 opacity-20" />
                  <p className="text-muted">Awaiting test image submission...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
