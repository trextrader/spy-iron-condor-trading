/**
 * Fuzzy Gate Heatmap
 * Phase 6.7 - Model Introspection
 *
 * Visualizes fuzzy gate activations over training time.
 * Shows which gates wake up, die, or oscillate.
 * This is "watching CondorNet think."
 */

import { useMemo, useRef, useEffect, useState } from 'react';

interface FuzzyGateHeatmapProps {
  steps: number[];
  gates: string[];
  values: number[][]; // [time_steps][gates]
  height?: number;
}

// Magma-inspired color scale
function getHeatmapColor(value: number): string {
  // Clamp to [0, 1]
  const v = Math.max(0, Math.min(1, value));

  // Magma palette approximation
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

export default function FuzzyGateHeatmap({
  steps,
  gates,
  values,
  height = 300,
}: FuzzyGateHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{
    step: number;
    gate: string;
    value: number;
  } | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height });

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
    if (!canvas || !values.length || !gates.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const numSteps = values.length;
    const numGates = gates.length;

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
      for (let g = 0; g < numGates; g++) {
        const value = values[t]?.[g] ?? 0;
        ctx.fillStyle = getHeatmapColor(value);
        ctx.fillRect(
          labelWidth + t * cellWidth,
          g * cellHeight,
          cellWidth + 0.5, // Slight overlap to prevent gaps
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
      ctx.fillText(gates[g] || `F${g}`, labelWidth - 5, g * cellHeight + cellHeight / 2);
    }

    // Draw step labels (every 1000 steps)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const labelInterval = Math.max(1, Math.floor(numSteps / 10));
    for (let t = 0; t < numSteps; t += labelInterval) {
      const step = steps[t] ?? t;
      ctx.fillText(
        `${(step / 1000).toFixed(0)}k`,
        labelWidth + t * cellWidth,
        plotHeight + 5
      );
    }
  }, [values, gates, steps, dimensions]);

  // Handle mouse move for tooltip
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !values.length) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const labelWidth = 40;
    const plotWidth = dimensions.width - labelWidth - 20;
    const plotHeight = dimensions.height - 40;

    const cellWidth = plotWidth / values.length;
    const cellHeight = plotHeight / gates.length;

    const stepIndex = Math.floor((x - labelWidth) / cellWidth);
    const gateIndex = Math.floor(y / cellHeight);

    if (
      stepIndex >= 0 &&
      stepIndex < values.length &&
      gateIndex >= 0 &&
      gateIndex < gates.length
    ) {
      setHoveredCell({
        step: steps[stepIndex] ?? stepIndex,
        gate: gates[gateIndex] ?? `F${gateIndex}`,
        value: values[stepIndex]?.[gateIndex] ?? 0,
      });
    } else {
      setHoveredCell(null);
    }
  };

  if (!values.length || !gates.length) {
    return (
      <div className="flex h-64 items-center justify-center text-surface-500">
        No fuzzy gate data available
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
            Step: <span className="text-surface-100">{hoveredCell.step}</span>
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
