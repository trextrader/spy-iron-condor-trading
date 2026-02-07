import { clsx } from 'clsx';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  fullScreen?: boolean;
  message?: string;
}

export default function LoadingSpinner({
  size = 'md',
  fullScreen = false,
  message,
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4 border-2',
    md: 'h-8 w-8 border-2',
    lg: 'h-12 w-12 border-3',
  };

  const spinner = (
    <div className="flex flex-col items-center justify-center">
      <div
        className={clsx(
          'animate-spin rounded-full border-brand-500 border-t-transparent',
          sizeClasses[size]
        )}
      />
      {message && <p className="mt-3 text-sm text-surface-400">{message}</p>}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="flex h-full min-h-[400px] items-center justify-center">{spinner}</div>
    );
  }

  return spinner;
}
