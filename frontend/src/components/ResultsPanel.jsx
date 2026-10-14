import { MODEL_CONFIGS } from '../config/models';

export const ResultsPanel = ({ result, previewUrl }) => {
  const isReal = result.prediction === 'REAL';
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const fakePercent = (result.probabilities.fake * 100).toFixed(1);
  const realPercent = (result.probabilities.real * 100).toFixed(1);
  const modelConfig = MODEL_CONFIGS[result.model] || {
    name: result.model,
    role: 'Model',
    shortDescription: '',
    detailedDescription: '',
    parametersEstimated: '',
  };

  return (
    <div className="space-y-6 pt-6 border-t border-slate-200">
      {/* Top Section: Verdict & Preview */}
      <div className="flex items-center justify-between pb-5 border-b border-slate-200">
        <div className="space-y-1.5">
          <span className="text-xs sm:text-sm uppercase tracking-wider text-slate-500 font-bold">
            Prediction Result
          </span>
          <div className="flex items-center space-x-4">
            <span
              className={`text-4xl sm:text-5xl font-bold tracking-tight ${
                isReal ? 'text-emerald-800' : 'text-rose-800'
              }`}
            >
              {result.prediction}
            </span>
            <span
              className={`text-sm sm:text-base px-3 py-1 font-bold border ${
                isReal
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : 'bg-rose-50 text-rose-800 border-rose-200'
              }`}
            >
              {confidencePercent}% confidence
            </span>
          </div>

          {/* Clean Inline Metadata */}
          <p className="text-sm sm:text-base text-slate-700 pt-1">
            Model: <span className="font-bold text-slate-900">{modelConfig.name}</span>
            <span className="mx-2 text-slate-400">•</span>
            Time: <span className="font-mono text-sm text-slate-900">{result.processing_time_ms.toFixed(0)} ms</span>
            <span className="mx-2 text-slate-400">•</span>
            Device: <span className="font-mono text-sm text-slate-900">{result.device_used?.toUpperCase() || 'CPU'}</span>
          </p>
        </div>

        {/* Thumbnail */}
        {previewUrl && (
          <img
            src={previewUrl}
            alt="Analyzed face"
            className="h-20 w-20 object-cover border border-slate-300"
          />
        )}
      </div>

      {/* Horizontal Probability Distribution */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm sm:text-base text-slate-700">
          <span>
            Fake Probability: <strong className="font-mono text-sm sm:text-base text-slate-900">{fakePercent}%</strong>
          </span>
          <span>
            Real Probability: <strong className="font-mono text-sm sm:text-base text-slate-900">{realPercent}%</strong>
          </span>
        </div>

        <div className="h-3 w-full bg-slate-200 overflow-hidden flex">
          <div
            style={{ width: `${result.probabilities.fake * 100}%` }}
            className="bg-rose-600 h-full transition-all duration-300"
            title={`Fake: ${fakePercent}%`}
          />
          <div
            style={{ width: `${result.probabilities.real * 100}%` }}
            className="bg-emerald-700 h-full transition-all duration-300"
            title={`Real: ${realPercent}%`}
          />
        </div>
      </div>

      {/* Respective Model Description (Appears only after analysis) */}
      <div className="pt-5 border-t border-slate-200 space-y-2">
        <h4 className="text-base sm:text-lg font-bold text-slate-900">
          Model Description: {modelConfig.name} ({modelConfig.role})
        </h4>
        <p className="text-sm sm:text-base text-slate-700 leading-relaxed max-w-4xl">
          {modelConfig.detailedDescription || modelConfig.shortDescription}
          {modelConfig.parametersEstimated
            ? ` The neural network architecture comprises approximately ${modelConfig.parametersEstimated.replace('~', '')}.`
            : ''}
        </p>
      </div>

      {/* Quiet Disclaimer */}
      <p className="text-xs sm:text-sm text-slate-500 border-t border-slate-200 pt-3 leading-relaxed italic">
        This prediction is an automated estimate from the selected model. It is not definitive proof that an image is real or fake.
      </p>
    </div>
  );
};
