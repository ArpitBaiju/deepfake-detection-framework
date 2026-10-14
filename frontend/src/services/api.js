const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export class ApiServiceError extends Error {
  constructor(message, statusCode, errorCode, details) {
    super(message);
    this.name = 'ApiServiceError';
    this.statusCode = statusCode;
    this.errorCode = errorCode;
    this.details = details;
  }
}

export const api = {
  /**
   * Health check and runtime hardware diagnostics.
   */
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });

      if (!response.ok) {
        throw new ApiServiceError(`Backend returned status ${response.status}`, response.status);
      }

      return await response.json();
    } catch (err) {
      if (err instanceof ApiServiceError) throw err;
      throw new ApiServiceError(
        'Inference service not connected. Please ensure the backend is running at ' + API_BASE_URL,
        0,
        'CONNECTION_REFUSED'
      );
    }
  },

  /**
   * Fetch backend models list and availability of .pth weights.
   */
  async getModels() {
    try {
      const response = await fetch(`${API_BASE_URL}/models`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });

      if (!response.ok) {
        throw new ApiServiceError(`Failed to fetch models (HTTP ${response.status})`, response.status);
      }

      const data = await response.json();
      return {
        models: data.models,
        weightsDirectory: data.weights_directory,
        mockMode: data.mock_mode,
      };
    } catch (err) {
      if (err instanceof ApiServiceError) throw err;
      throw new ApiServiceError('Cannot connect to backend models registry.', 0, 'CONNECTION_REFUSED');
    }
  },

  /**
   * Upload image and perform deepfake classification.
   */
  async predictImage(file, modelId) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', modelId);

    let response;
    try {
      response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
    } catch (err) {
      throw new ApiServiceError(
        'Inference service not connected. Please ensure the backend server is running.',
        0,
        'SERVICE_DISCONNECTED',
        err
      );
    }

    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch {
        errorData = { detail: `HTTP ${response.status}: ${response.statusText}` };
      }

      const detail = errorData.detail || errorData;
      if (response.status === 503 && typeof detail === 'object' && detail.error_code === 'WEIGHTS_NOT_FOUND') {
        throw new ApiServiceError(
          detail.message || `Model weights for '${modelId}' are not loaded on the backend.`,
          503,
          'WEIGHTS_NOT_FOUND',
          detail
        );
      }

      const errorMsg = typeof detail === 'string' ? detail : detail.message || JSON.stringify(detail);
      throw new ApiServiceError(errorMsg, response.status, errorData.error_code || 'PREDICT_FAILED', detail);
    }

    const json = await response.json();

    const result = {
      id: 'pred-' + Date.now() + '-' + Math.random().toString(36).substring(2, 7),
      prediction: json.prediction,
      class_id: json.class_id,
      confidence: json.confidence,
      model: json.model,
      processing_time_ms: json.processing_time_ms,
      probabilities: json.probabilities || {
        fake: json.prediction === 'FAKE' ? json.confidence : 1 - json.confidence,
        real: json.prediction === 'REAL' ? json.confidence : 1 - json.confidence,
      },
      is_mock: json.is_mock || false,
      device_used: json.device_used,
      timestamp: new Date().toISOString(),
      imageFileName: file.name,
      imageFileSize: file.size,
    };

    return result;
  },
};
