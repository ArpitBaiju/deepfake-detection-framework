export const Footer = () => {
  return (
    <footer className="border-t border-slate-200 mt-16 py-8 text-xs text-slate-500">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <p>
          Deepfake Detection Framework • Research and verification utility
        </p>
        <p className="text-slate-500 font-mono text-[11px]">
          Standard: 256×256 RGB • Fake = 0, Real = 1
        </p>
      </div>
    </footer>
  );
};
