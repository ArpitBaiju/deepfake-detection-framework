import { Shield } from 'lucide-react';

export const Header = ({
  backendHealth,
  backendError,
  onRefreshHealth,
}) => {
  const isConnected = !!backendHealth && !backendError;

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <Shield className="h-6 w-6 text-blue-600 shrink-0" />
          <span className="font-bold text-lg sm:text-xl text-slate-900 tracking-normal">
            Deepfake Detection Framework
          </span>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onRefreshHealth}
            title="Click to check backend status"
            className="flex items-center space-x-2 text-sm text-slate-700 hover:text-slate-900 transition-colors px-3 py-1.5 hover:bg-slate-50"
          >
            <span
              className={`h-2.5 w-2.5 ${
                isConnected ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            <span className="font-medium">
              {isConnected
                ? `Backend connected (${backendHealth?.device.toUpperCase()})`
                : 'Backend disconnected'}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
};
