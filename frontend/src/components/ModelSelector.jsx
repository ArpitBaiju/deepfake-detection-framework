import { Loader2 } from 'lucide-react';
import { MODEL_LIST } from '../config/models';

export const ModelSelector = ({
  selectedModelId,
  onSelectModel,
  onAnalyze,
  isAnalyzing,
  canAnalyze,
  inferenceError,
  backendConnected,
}) => {
  return (
    <div className="space-y-4">
      <div>
        <label className="text-base sm:text-lg font-bold text-slate-900 block">
          2. Select Model
        </label>
        <span className="text-sm text-slate-600">
          Choose which model will analyze the image
        </span>
      </div>

      {/* Clean, unboxed radio-style list */}
      <div className="space-y-2">
        {MODEL_LIST.map((model) => {
          const isSelected = selectedModelId === model.id;

          return (
            <label
              key={model.id}
              onClick={() => !isAnalyzing && onSelectModel(model.id)}
              className={`flex items-start space-x-3 p-3 cursor-pointer transition-colors ${
                isSelected
                  ? 'bg-blue-50/80 text-slate-900'
                  : 'hover:bg-slate-50 text-slate-700'
              } ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <input
                type="radio"
                name="model-selection"
                checked={isSelected}
                onChange={() => !isAnalyzing && onSelectModel(model.id)}
                disabled={isAnalyzing}
                className="mt-1 h-4 w-4 text-blue-600 border-slate-300 focus:ring-blue-500"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className={`text-base font-bold ${isSelected ? 'text-blue-950' : 'text-slate-900'}`}>
                    {model.name}
                  </span>
                  <span className="text-xs sm:text-sm text-slate-600 italic">
                    {model.role}
                  </span>
                </div>
                <p className="text-sm text-slate-600 mt-1 leading-relaxed">
                  {model.shortDescription}
                </p>
              </div>
            </label>
          );
        })}
      </div>

      {/* Notices */}
      {!backendConnected && (
        <p className="text-sm text-rose-700 bg-rose-50 border border-rose-200 p-3">
          Inference service not connected. Please ensure the backend is running.
        </p>
      )}

      {inferenceError && (
        <p className="text-sm text-amber-800 bg-amber-50 border border-amber-200 p-3">
          {inferenceError}
        </p>
      )}

      {/* Analyze Button */}
      <div className="pt-2">
        <button
          onClick={onAnalyze}
          disabled={!canAnalyze || isAnalyzing || !backendConnected}
          className={`w-full py-3 px-5 font-bold text-base transition-colors flex items-center justify-center space-x-2 ${
            canAnalyze && !isAnalyzing && backendConnected
              ? 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white cursor-pointer shadow-sm'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
          }`}
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin text-slate-500" />
              <span>Analyzing image...</span>
            </>
          ) : (
            <span>Analyze Image</span>
          )}
        </button>

        {!canAnalyze && (
          <p className="text-center text-sm text-slate-600 mt-2">
            Upload an image to start analysis
          </p>
        )}
      </div>
    </div>
  );
};
