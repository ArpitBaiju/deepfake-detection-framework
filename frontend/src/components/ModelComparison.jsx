import { MODEL_LIST } from '../config/models';

export const ModelComparison = () => {
  return (
    <section className="space-y-4">
      <div>
        <h3 className="text-base font-semibold text-slate-900">
          Model Overview
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Four standard architectures configured for binary classification (Fake = 0, Real = 1).
        </p>
      </div>

      {/* 4 Compact Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {MODEL_LIST.map((model) => (
          <div
            key={model.id}
            className="p-4 bg-white border border-slate-200 space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-900">
                {model.name}
              </span>
              <span className="text-[11px] text-slate-500 font-mono">
                {model.parametersEstimated}
              </span>
            </div>

            <p className="text-xs text-slate-600 leading-normal">
              {model.shortDescription}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
};
