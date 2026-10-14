import { useRef, useState, useCallback } from 'react';
import { Upload, AlertCircle } from 'lucide-react';

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];

export const ImageUpload = ({
  selectedFile,
  previewUrl,
  onImageSelected,
  onImageRemoved,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [imageMeta, setImageMeta] = useState(null);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = useCallback(
    (file) => {
      setErrorMessage(null);

      const isValidType =
        ALLOWED_TYPES.includes(file.type.toLowerCase()) ||
        /\.(jpe?g|png)$/i.test(file.name);

      if (!isValidType) {
        setErrorMessage(`Invalid format: "${file.name}". Supported: JPG, JPEG, PNG.`);
        return;
      }

      if (file.size > MAX_FILE_SIZE_BYTES) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        setErrorMessage(`File exceeds 10 MB limit (${sizeMb} MB).`);
        return;
      }

      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        setImageMeta({ width: img.naturalWidth, height: img.naturalHeight });
        onImageSelected(file, url);
      };
      img.onerror = () => {
        setErrorMessage('Unable to decode image file.');
        URL.revokeObjectURL(url);
      };
      img.src = url;
    },
    [onImageSelected]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleRemove = () => {
    setImageMeta(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onImageRemoved();
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-base sm:text-lg font-bold text-slate-900 block">
            1. Facial Image
          </label>
          <span className="text-sm text-slate-600">
            JPG, JPEG, or PNG (up to 10 MB)
          </span>
        </div>
        {selectedFile && (
          <button
            onClick={handleRemove}
            disabled={disabled}
            className="text-sm text-slate-600 hover:text-slate-900 underline transition-colors"
          >
            Change image
          </button>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        onChange={handleFileChange}
        disabled={disabled}
        className="hidden"
        id="image-upload-input"
      />

      {previewUrl && selectedFile ? (
        /* Unboxed Preview */
        <div className="space-y-2">
          <div className="w-full bg-slate-100 overflow-hidden flex items-center justify-center h-72 border border-slate-200">
            <img
              src={previewUrl}
              alt="Uploaded face preview"
              className="max-h-64 w-auto max-w-full object-contain"
            />
          </div>
          <div className="flex items-center justify-between text-sm text-slate-600">
            <span className="truncate max-w-[240px] text-slate-800 font-medium">{selectedFile.name}</span>
            <span className="font-mono text-xs">
              {imageMeta ? `${imageMeta.width} × ${imageMeta.height} px • ` : ''}
              {formatFileSize(selectedFile.size)}
            </span>
          </div>
        </div>
      ) : (
        /* Dropzone */
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={`border border-dashed p-10 text-center cursor-pointer transition-colors ${
            isDragging
              ? 'border-blue-500 bg-blue-50/40'
              : 'border-slate-300 hover:border-slate-400 bg-slate-50/50 hover:bg-slate-50'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <Upload className="h-7 w-7 text-slate-500 mx-auto mb-2" />
          <p className="text-base font-semibold text-slate-800 mb-1">
            Click or drag image here
          </p>
          <p className="text-sm text-slate-600">
            Select a facial photograph to analyze
          </p>
        </div>
      )}

      {errorMessage && (
        <div className="p-3 bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center space-x-2">
          <AlertCircle className="h-4 w-4 text-rose-600 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
