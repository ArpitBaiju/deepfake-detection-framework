import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ImageUpload } from './components/ImageUpload';
import { ModelSelector } from './components/ModelSelector';
import { ResultsPanel } from './components/ResultsPanel';
import { HistoryPanel } from './components/HistoryPanel';
import { Footer } from './components/Footer';
import { api, ApiServiceError } from './services/api';
import { useHistory } from './hooks/useHistory';
import { DEFAULT_MODEL_ID } from './config/models';

export const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState(DEFAULT_MODEL_ID);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [inferenceError, setInferenceError] = useState(null);

  const [backendHealth, setBackendHealth] = useState(null);
  const [backendError, setBackendError] = useState(null);
  const [backendModels, setBackendModels] = useState([]);

  const { history, addHistoryItem, removeHistoryItem, clearHistory } = useHistory();

  const fetchBackendStatus = useCallback(async () => {
    try {
      const health = await api.checkHealth();
      setBackendHealth(health);
      setBackendError(null);

      const modelsData = await api.getModels();
      setBackendModels(modelsData.models);
    } catch (err) {
      setBackendError(err.message || 'Could not connect to inference backend.');
      setBackendHealth(null);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const checkStatus = async () => {
      try {
        const health = await api.checkHealth();
        if (!isMounted) return;
        setBackendHealth(health);
        setBackendError(null);

        const modelsData = await api.getModels();
        if (!isMounted) return;
        setBackendModels(modelsData.models);
      } catch (err) {
        if (!isMounted) return;
        setBackendError(err.message || 'Could not connect to inference backend.');
        setBackendHealth(null);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleImageSelected = useCallback((file, url) => {
    setSelectedFile(file);
    setPreviewUrl(url);
    setPredictionResult(null);
    setInferenceError(null);
  }, []);

  const handleImageRemoved = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setPredictionResult(null);
    setInferenceError(null);
  }, [previewUrl]);

  const handleAnalyze = async () => {
    if (!selectedFile || !selectedModelId) return;

    setIsAnalyzing(true);
    setInferenceError(null);

    try {
      const result = await api.predictImage(selectedFile, selectedModelId);
      setPredictionResult(result);
      addHistoryItem(result, previewUrl || undefined);

      setTimeout(() => {
        const resultsEl = document.getElementById('results-section');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err) {
      if (err instanceof ApiServiceError && err.errorCode === 'WEIGHTS_NOT_FOUND') {
        setInferenceError(err.message);
      } else {
        setInferenceError(err.message || 'Failed to complete model inference.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const canAnalyze = !!selectedFile && !!selectedModelId;
  const backendConnected = !!backendHealth && !backendError;

  return (
    <div className="min-h-screen flex flex-col justify-between bg-slate-50 text-slate-900 font-serif">
      {/* Header */}
      <Header
        backendHealth={backendHealth}
        backendError={backendError}
        onRefreshHealth={fetchBackendStatus}
      />

      {/* Main Container - Full Widescreen Width */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 space-y-10 flex-1 w-full">
        {/* Simple Page Heading */}
        <div className="space-y-1.5">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
            Deepfake Detection Framework
          </h2>
          <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-4xl">
            Analyze an image using trained deep-learning models and review the prediction with appropriate caution.
          </p>
        </div>

        {/* Unified, Single-Card Workflow */}
        <section className="bg-white border border-slate-200 p-6 sm:p-8 shadow-sm">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Step 1: Upload */}
            <div className="lg:col-span-6">
              <ImageUpload
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onImageSelected={handleImageSelected}
                onImageRemoved={handleImageRemoved}
                disabled={isAnalyzing}
              />
            </div>

            {/* Step 2: Model Selection & Analyze Action */}
            <div className="lg:col-span-6 lg:border-l lg:border-slate-200 lg:pl-8">
              <ModelSelector
                selectedModelId={selectedModelId}
                onSelectModel={setSelectedModelId}
                backendModels={backendModels}
                onAnalyze={handleAnalyze}
                isAnalyzing={isAnalyzing}
                canAnalyze={canAnalyze}
                inferenceError={inferenceError}
                backendConnected={backendConnected}
              />
            </div>
          </div>
        </section>

        {/* Results Section (appears only when output is obtained) */}
        {predictionResult && (
          <section id="results-section">
            <ResultsPanel result={predictionResult} previewUrl={previewUrl} />
          </section>
        )}

        {/* History Section */}
        <HistoryPanel
          history={history}
          onClearHistory={clearHistory}
          onRemoveItem={removeHistoryItem}
        />
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default App;
