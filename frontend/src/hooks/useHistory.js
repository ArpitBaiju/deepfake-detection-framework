import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'deepfake_detection_history_v1';
const MAX_HISTORY_ITEMS = 25;

export function useHistory() {
  const [history, setHistory] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (err) {
      console.error('Failed to parse history from localStorage', err);
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (err) {
      console.warn('Failed to save history to localStorage (possibly quota exceeded)', err);
    }
  }, [history]);

  const addHistoryItem = useCallback((result, thumbnailDataUrl) => {
    const formatBytes = (bytes) => {
      if (!bytes) return '';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const newItem = {
      id: result.id,
      thumbnail: thumbnailDataUrl || result.imagePreviewUrl || '',
      prediction: result.prediction,
      confidence: result.confidence,
      model: result.model,
      timestamp: result.timestamp,
      processingTimeMs: result.processing_time_ms,
      fileName: result.imageFileName || 'uploaded-image.jpg',
      fileSizeFormatted: formatBytes(result.imageFileSize),
    };

    setHistory((prev) => [newItem, ...prev.slice(0, MAX_HISTORY_ITEMS - 1)]);
  }, []);

  const removeHistoryItem = useCallback((id) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      console.error('Failed to clear history from localStorage', err);
    }
  }, []);

  return {
    history,
    addHistoryItem,
    removeHistoryItem,
    clearHistory,
  };
}
