import { Trash2, X } from 'lucide-react';
import { MODEL_CONFIGS } from '../config/models';

export const HistoryPanel = ({
  history,
  onClearHistory,
  onRemoveItem,
}) => {
  const formatDate = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <section className="space-y-4 pt-8 border-t border-slate-200">
      <div className="flex items-center justify-between">
        <h3 className="text-base sm:text-lg font-bold text-slate-900">
          Recent Analyses
        </h3>

        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="text-sm text-slate-600 hover:text-slate-900 flex items-center space-x-1.5 transition-colors"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Clear history</span>
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <p className="text-sm text-slate-600 py-2">
          No past analyses recorded in this session.
        </p>
      ) : (
        <div className="divide-y divide-slate-200">
          {history.map((item) => {
            const isReal = item.prediction === 'REAL';
            const modelName = MODEL_CONFIGS[item.model]?.name || item.model;

            return (
              <div
                key={item.id}
                className="py-3 flex items-center justify-between hover:bg-slate-100/50 px-3 transition-colors text-sm"
              >
                {/* Thumbnail & File Details */}
                <div className="flex items-center space-x-3.5 min-w-0">
                  <div className="h-10 w-10 bg-slate-100 border border-slate-300 overflow-hidden shrink-0">
                    {item.thumbnail ? (
                      <img
                        src={item.thumbnail}
                        alt="Thumbnail"
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="h-full w-full bg-slate-200" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className="font-bold text-slate-900 truncate max-w-[160px] sm:max-w-[320px]" title={item.fileName}>
                      {item.fileName}
                    </p>
                    <span className="text-xs text-slate-600">
                      {formatDate(item.timestamp)} • {modelName}
                    </span>
                  </div>
                </div>

                {/* Verdict, Confidence & Delete */}
                <div className="flex items-center space-x-4 shrink-0">
                  <span
                    className={`font-bold px-2.5 py-0.5 text-xs border ${
                      isReal
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : 'bg-rose-50 text-rose-800 border-rose-200'
                    }`}
                  >
                    {item.prediction} ({(item.confidence * 100).toFixed(0)}%)
                  </span>

                  <button
                    onClick={() => onRemoveItem(item.id)}
                    title="Remove item"
                    className="text-slate-400 hover:text-slate-700 p-1"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};
