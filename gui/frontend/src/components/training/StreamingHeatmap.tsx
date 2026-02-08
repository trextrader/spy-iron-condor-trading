/**
 * Streaming Heatmap
 * Phase 6.8 - Real-time Training Visualization
 *
 * Real-time fuzzy gate activation heatmap.
 */

import { useMemo, useRef, useEffect, useState } from 'react';
import { FuzzyActivations } from '@/hooks/useTrainingTelemetry';

interface StreamingHeatmapProps {
  fuzzyHistory: FuzzyActivations[];
  gateLabels?: string[];
  visibleWindow?: number;
  height?: number;
}

// Magma-inspired color scale
function getHeatmapColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));

  if (v < 0.25) {
    const t = v / 0.25;
    return `rgb(${Math.floor(t * 80)}, ${Math.floor(t * 20)}, ${Math.floor(60 + t * 80)})`;
  } else if (v < 0.5) {
    const t = (v - 0.25) / 0.25;
    return `rgb(${Math.floor(80 + t * 120)}, ${Math.floor(20 + t * 30)}, ${Math.floor(140 - t * 40)})`;
  } else if (v < 0.75) {
    const t = (v - 0.5) / 0.25;
    return `rgb(${Math.floor(200 + t * 50)}, ${Math.floor(50 + t * 100)}, ${Math.floor(100 - t * 50)})`;
  } else {
    const t = (v - 0.75) / 0.25;
    return `rgb(${Math.floor(250 + t * 5)}, ${Math.floor(150 + t * 100)}, ${Math.floor(50 + t * 150)})`;
  }
}

export default function StreamingHeatmap({
  fuzzyHistory,
  gateLabels,
  visibleWindow = 200,
  height = 250,
}: StreamingHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height });
  const [hoveredCell, setHoveredCell] = useState<{
    step: number;
    gate: string;
    value: number;
  } | null>(null);

  // Get visible data slice
  const visibleData = useMemo(() => {
    return fuzzyHistory.slice(-visibleWindow);
  }, [fuzzyHistory, visibleWindow]);

  // Determine number of gates from data
  const numGates = visibleData[0]?.activations.length ?? 10;
  const gates = gateLabels ?? Array.from({ length: numGates }, (_, i) => `F${i}`);

  // Handle resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height,
        });
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [height]);

  // Draw heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !visibleData.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const numSteps = visibleData.length;

    // Calculate cell dimensions
    const labelWidth = 40;
    const plotWidth = dimensions.width - labelWidth - 20;
    const plotHeight = dimensions.height - 40;

    const cellWidth = plotWidth / numSteps;
    const cellHeight = plotHeight / numGates;

    // Clear canvas
    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    // Draw heatmap cells
    for (let t = 0; t < numSteps; t++) {
      const activations = visibleData[t]?.activations ?? [];
      for (let g = 0; g < numGates; g++) {
        const value = activations[g] ?? 0;
        ctx.fillStyle = getHeatmapColor(value);
        ctx.fillRect(
          labelWidth + t * cellWidth,
          g * cellHeight,
          cellWidth + 0.5,
          cellHeight + 0.5
        );
      }
    }

    // Draw gate labels
    ctx.fillStyle = '#a1a1aa';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let g = 0; g < numGates; g++) {
      ctx.fillText(gates[g], labelWidth - 5, g * cellHeight + cellHeight / 2);
    }

    // Draw step labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const labelInterval = Math.max(1, Math.floor(numSteps / 8));
    for (let t = 0; t < numSteps; t += labelInterval) {
      const step = visibleData[t]?.step ?? t;
      ctx.fillText(
        `${(step / 1000).toFixed(1)}k`,
        labelWidth + t * cellWidth + cellWidth / 2,
        plotHeight + 5
      );
    }
  }, [visibleData, gates, numGates, dimensions]);

  // Handle mouse move for tooltip
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !visibleData.length) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const labelWidth = 40;
    const plotWidth = dimensions.width - labelWidth - 20;
    const plotHeight = dimensions.height - 40;

    const cellWidth = plotWidth / visibleData.length;
    const cellHeight = plotHeight / numGates;

    const stepIndex = Math.floor((x - labelWidth) / cellWidth);
    const gateIndex = Math.floor(y / cellHeight);

    if (
      stepIndex >= 0 &&
      stepIndex < visibleData.length &&
      gateIndex >= 0 &&
      gateIndex < numGates
    ) {
      setHoveredCell({
        step: visibleData[stepIndex]?.step ?? stepIndex,
        gate: gates[gateIndex],
        value: visibleData[stepIndex]?.activations[gateIndex] ?? 0,
      });
    } else {
      setHoveredCell(null);
    }
  };

  if (!visibleData.length) {
    return (
      <div className="flex items-center justify-center text-surface-500" style={{ height }}>
        <div className="text-center">
          <p className="text-lg">Waiting for fuzzy gate data...</p>
          <p className="mt-1 text-sm">Heatmap will populate as training progresses</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredCell(null)}
        className="cursor-crosshair"
      />

      {/* Tooltip */}
      {hoveredCell && (
        <div
          className="pointer-events-none absolute rounded-lg border border-surface-700 bg-surface-900 px-3 py-2 text-sm shadow-lg"
          style={{ top: 10, right: 10 }}
        >
          <div className="text-surface-400">
            Step: <span className="text-surface-100">{hoveredCell.step.toLocaleString()}</span>
          </div>
          <div className="text-surface-400">
            Gate: <span className="text-surface-100">{hoveredCell.gate}</span>
          </div>
          <div className="text-surface-400">
            Activation:{' '}
            <span
              className="font-mono"
              style={{ color: getHeatmapColor(hoveredCell.value) }}
            >
              {hoveredCell.value.toFixed(4)}
            </span>
          </div>
        </div>
      )}

      {/* Color scale legend */}
      <div className="mt-2 flex items-center justify-center gap-2">
        <span className="text-xs text-surface-500">0.0</span>
        <div
          className="h-3 w-48 rounded"
          style={{
            background: `linear-gradient(to right,
              ${getHeatmapColor(0)},
              ${getHeatmapColor(0.25)},
              ${getHeatmapColor(0.5)},
              ${getHeatmapColor(0.75)},
              ${getHeatmapColor(1)}
            )`,
          }}
        />
        <span className="text-xs text-surface-500">1.0</span>
      </div>
    </div>
  );
}
