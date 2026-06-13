import React, { useState, useEffect, useRef } from "react";
import { uploadDataset, getDatasets, deleteDataset } from "../services/api";
import { Upload, FileArchive, CheckCircle, AlertTriangle, ArrowRight, RefreshCw, Database, Trash2, Play } from "lucide-react";

export default function DatasetUpload({ setActiveTab, setDatasetContextId }) {
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const [datasets, setDatasets] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [delLoadingId, setDelLoadingId] = useState(null);
  const [listLoading, setListLoading] = useState(true);

  const fetchDatasetsList = async () => {
    try {
      setListLoading(true);
      const data = await getDatasets();
      setDatasets(data);
      setLoadError(null);
    } catch (err) {
      console.error(err);
      setLoadError("Failed to fetch uploaded datasets list.");
    } finally {
      setListLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasetsList();
  }, []);

  const handleDeleteDataset = async (datasetId) => {
    if (!window.confirm("Are you sure you want to delete this dataset? This will also cascade delete all associated training runs and models.")) {
      return;
    }

    try {
      setDelLoadingId(datasetId);
      await deleteDataset(datasetId);
      await fetchDatasetsList();
      // If we deleted the active context dataset, reset it
      setDatasetContextId(null);
    } catch (err) {
      console.error(err);
      alert(err.message || "Failed to delete dataset.");
    } finally {
      setDelLoadingId(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith(".zip")) {
        setFile(droppedFile);
        if (!name) {
          setName(droppedFile.name.replace(".zip", ""));
        }
      } else {
        setError("Only ZIP format files are supported.");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      if (!name) {
        setName(selectedFile.name.replace(".zip", ""));
      }
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Please provide a name for this dataset.");
      return;
    }
    if (!file) {
      setError("Please select a ZIP dataset file.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const data = await uploadDataset(name, file);
      setResult(data);
      if (data.is_validated) {
        setDatasetContextId(data.id);
      }
      fetchDatasetsList();
    } catch (err) {
      setError(err.message || "An unexpected error occurred during upload.");
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setName("");
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="container py-4" style={{ maxWidth: "800px" }}>
      <div className="glass-panel">
        <div className="glass-header">
          <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
            <Upload size={18} className="text-indigo" />
            Upload YOLO Training Dataset
          </h5>
        </div>

        <div className="p-4">
          {!result && !loading && (
            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <label className="form-label text-muted small fw-semibold">Dataset Identifier</label>
                <input
                  type="text"
                  placeholder="e.g. container_numbers_v1"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="form-control form-control-custom"
                  required
                />
              </div>

              <div className="mb-4">
                <label className="form-label text-muted small fw-semibold">Upload File (ZIP format)</label>
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={triggerFileSelect}
                  className={`dropzone-area ${dragOver ? "drag-over" : ""}`}
                >
                  <input
                    type="file"
                    accept=".zip"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    style={{ display: "none" }}
                  />
                  <div className="d-flex flex-column align-items-center">
                    <FileArchive size={48} className={file ? "text-indigo mb-3" : "text-muted mb-3 opacity-50"} />
                    {file ? (
                      <div>
                        <span className="fw-semibold text-white d-block">{file.name}</span>
                        <span className="text-muted small">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                      </div>
                    ) : (
                      <div>
                        <span className="fw-semibold text-white d-block">Drag & drop dataset ZIP file here</span>
                        <span className="text-muted small">or click to browse local files</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {error && (
                <div className="alert alert-danger bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-danger small">
                  <strong>Error:</strong> {error}
                </div>
              )}

              <button
                type="submit"
                className="btn btn-indigo w-100"
                disabled={!name || !file}
              >
                Upload & Validate Dataset
              </button>
            </form>
          )}

          {loading && (
            <div className="text-center py-5">
              <RefreshCw size={48} className="text-indigo mb-4 spin-animation" style={{ animation: "spin 1.5s linear infinite" }} />
              <h5 className="fw-bold text-white mb-2">Uploading and Validating Dataset...</h5>
              <p className="text-muted small">Extracting ZIP, parsing data.yaml, matching images/labels, and validating coordinate ranges.</p>
              
              <style>{`
                @keyframes spin {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
              `}</style>
            </div>
          )}

          {result && (
            <div className="text-center py-4">
              {result.is_validated ? (
                <div>
                  <div className="p-3 bg-teal-opacity rounded-circle d-inline-flex mb-3" style={{ backgroundColor: "rgba(20, 184, 166, 0.1)" }}>
                    <CheckCircle size={48} className="text-teal" />
                  </div>
                  <h4 className="fw-bold text-white mb-2">Dataset Validation Passed!</h4>
                  <p className="text-muted mb-4">{result.validation_message}</p>
                  
                  <div className="glass-panel p-3 mb-4 text-start d-flex justify-content-around bg-dark-opacity">
                    <div>
                      <span className="text-muted small d-block">Images Found</span>
                      <span className="fw-bold text-white">{result.num_images}</span>
                    </div>
                    <div style={{ width: "1px", background: "var(--glass-border)" }}></div>
                    <div>
                      <span className="text-muted small d-block">Labels Verified</span>
                      <span className="fw-bold text-white">{result.num_labels}</span>
                    </div>
                  </div>

                  <div className="d-flex gap-2">
                    <button onClick={resetUpload} className="btn btn-glass flex-grow-1">
                      Upload Another
                    </button>
                    <button 
                      onClick={() => setActiveTab("train")} 
                      className="btn btn-indigo flex-grow-1 d-flex align-items-center justify-content-center gap-2"
                    >
                      Start Training Run <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="p-3 bg-rose-opacity rounded-circle d-inline-flex mb-3" style={{ backgroundColor: "rgba(244, 63, 94, 0.1)" }}>
                    <AlertTriangle size={48} className="text-rose" />
                  </div>
                  <h4 className="fw-bold text-white mb-2">Dataset Validation Failed</h4>
                  <p className="text-rose mb-4">{result.validation_message}</p>
                  
                  <div className="glass-panel p-3 mb-4 text-start bg-dark-opacity">
                    <span className="text-muted small d-block mb-1">Error Diagnostics</span>
                    <p className="text-white small mb-0 font-monospace" style={{ wordBreak: "break-all" }}>
                      {result.validation_message}
                    </p>
                  </div>

                  <button onClick={resetUpload} className="btn btn-indigo w-100">
                    Try Again
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Uploaded Datasets List */}
      <div className="glass-panel mt-5">
        <div className="glass-header d-flex justify-content-between align-items-center">
          <h5 className="mb-0 fw-bold text-white d-flex align-items-center gap-2">
            <Database size={18} className="text-indigo" />
            Uploaded Datasets Registry
          </h5>
          <button 
            type="button" 
            onClick={fetchDatasetsList} 
            className="btn btn-glass btn-sm d-flex align-items-center gap-1"
            disabled={listLoading}
          >
            <RefreshCw size={12} className={listLoading ? "spin-animation" : ""} /> Refresh
          </button>
        </div>

        <div className="p-4">
          {loadError && (
            <div className="alert alert-danger bg-dark-opacity border-0 p-3 mb-4 rounded-3 text-danger small">
              {loadError}
            </div>
          )}

          {listLoading && datasets.length === 0 ? (
            <div className="text-center py-4">
              <div className="spinner-border text-indigo spinner-border-sm" role="status">
                <span className="visually-hidden">Loading datasets...</span>
              </div>
            </div>
          ) : datasets.length === 0 ? (
            <div className="text-center py-4 text-muted small">
              No datasets uploaded yet. Upload a dataset ZIP above to begin.
            </div>
          ) : (
            <div className="table-responsive">
              <table className="table table-dark table-hover align-middle border-secondary mb-0 small">
                <thead>
                  <tr>
                    <th scope="col">ID</th>
                    <th scope="col">Name</th>
                    <th scope="col" className="text-center">Images</th>
                    <th scope="col" className="text-center">Labels</th>
                    <th scope="col" className="text-center">Validation</th>
                    <th scope="col" className="text-end">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((ds) => (
                    <tr key={ds.id}>
                      <td className="fw-bold text-muted">#{ds.id}</td>
                      <td className="fw-semibold text-white">{ds.name}</td>
                      <td className="text-center text-white">{ds.num_images}</td>
                      <td className="text-center text-white">{ds.num_labels}</td>
                      <td className="text-center">
                        {ds.is_validated ? (
                          <span className="badge bg-teal-opacity text-teal p-1 px-2 rounded-pill" style={{ fontSize: '0.75rem' }}>
                            Passed
                          </span>
                        ) : (
                          <span className="badge bg-rose-opacity text-rose p-1 px-2 rounded-pill" style={{ fontSize: '0.75rem' }}>
                            Failed
                          </span>
                        )}
                      </td>
                      <td className="text-end">
                        <div className="d-inline-flex gap-2">
                          {ds.is_validated && (
                            <button
                              onClick={() => {
                                setDatasetContextId(ds.id);
                                setActiveTab("train");
                              }}
                              className="btn btn-indigo btn-sm py-1 px-2 d-flex align-items-center gap-1"
                              title="Setup Training Run"
                            >
                              <Play size={12} /> Train
                            </button>
                          )}
                          <button
                            onClick={() => handleDeleteDataset(ds.id)}
                            className="btn btn-glass btn-sm py-1 px-2 text-rose border-rose-hover"
                            style={{ transition: 'all 0.2s' }}
                            disabled={delLoadingId === ds.id}
                            title="Delete Dataset"
                          >
                            {delLoadingId === ds.id ? (
                              <RefreshCw size={12} className="spin-animation" />
                            ) : (
                              <Trash2 size={12} />
                            )}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
