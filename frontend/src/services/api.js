const API_BASE_URL = window.location.origin.includes("localhost:5173")
  ? "http://localhost:8000"
  : window.location.origin;

export const getDashboardStats = async () => {
  const response = await fetch(`${API_BASE_URL}/api/dashboard/stats`);
  if (!response.ok) {
    throw new Error("Failed to fetch dashboard stats");
  }
  return response.json();
};

export const uploadDataset = async (name, file) => {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/datasets/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown upload error" }));
    throw new Error(errorData.detail || "Failed to upload dataset");
  }
  return response.json();
};

export const getDatasets = async () => {
  const response = await fetch(`${API_BASE_URL}/api/datasets`);
  if (!response.ok) {
    throw new Error("Failed to fetch datasets list");
  }
  return response.json();
};

export const deleteDataset = async (datasetId) => {
  const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Failed to delete dataset" }));
    throw new Error(errorData.detail || "Failed to delete dataset");
  }
  return response.json();
};

export const startTraining = async (datasetId, epochs, batchSize, imgsz, baseModel) => {
  const response = await fetch(`${API_BASE_URL}/api/training/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataset_id: datasetId,
      epochs,
      batch_size: batchSize,
      imgsz,
      base_model: baseModel,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown training start error" }));
    throw new Error(errorData.detail || "Failed to start training");
  }
  return response.json();
};

export const getTrainingStatus = async (jobId) => {
  const response = await fetch(`${API_BASE_URL}/api/training/status/${jobId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch training status");
  }
  return response.json();
};

export const getRunDetails = async (runId) => {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch run details");
  }
  return response.json();
};

export const getModels = async () => {
  const response = await fetch(`${API_BASE_URL}/api/models`);
  if (!response.ok) {
    throw new Error("Failed to fetch models list");
  }
  return response.json();
};

export const activateModel = async (modelId) => {
  const response = await fetch(`${API_BASE_URL}/api/models/${modelId}/activate`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error("Failed to activate model");
  }
  return response.json();
};

export const deleteModel = async (modelId) => {
  const response = await fetch(`${API_BASE_URL}/api/models/${modelId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Failed to delete model" }));
    throw new Error(errorData.detail || "Failed to delete model");
  }
  return response.json();
};

export const runInference = async (modelId, file) => {
  const formData = new FormData();
  formData.append("file", file);
  if (modelId) {
    formData.append("model_id", modelId);
  }

  const response = await fetch(`${API_BASE_URL}/api/inference/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown inference error" }));
    throw new Error(errorData.detail || "Inference failed");
  }
  return response.json();
};




