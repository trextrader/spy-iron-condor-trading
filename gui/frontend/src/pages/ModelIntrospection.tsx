/**
 * Model Introspection Page
 * Phase 6.7 - Neural Telemetry Dashboard
 *
 * The neural oscilloscope - watch CondorNet's brain form in real time.
 *
 * Features:
 * - Loss component trajectories (12-line heartbeat)
 * - Fuzzy gate activation heatmap (steps × gates)
 * - Gate distribution analysis (crisp vs soft logic)
 * - Epoch summary metrics
 */

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import LossTrajectoryChart from '@/components/introspection/LossTrajectoryChart';
import FuzzyGateHeatmap from '@/components/introspection/FuzzyGateHeatmap';
import GateDistributionPanel from '@/components/introspection/GateDistributionPanel';
import EpochSummaryPanel from '@/components/introspection/EpochSummaryPanel';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import {
  listDiagnostics,
  getLossTrajectory,
  getFuzzyHeatmap,
  getEpochSummary,
  getGateDistribution,
  type DiagnosticsFile,
} from '@/api/training';

type TabType = 'trajectories' | 'heatmap' | 'distributions' | 'summary';

export default function ModelIntrospection() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('trajectories');
  const [selectedEpoch, setSelectedEpoch] = useState<number>(1);
  const [smoothingWindow, setSmoothingWindow] = useState(200);

  // Fetch available diagnostics files
  const { data: filesData, isLoading: filesLoading } = useQuery({
    queryKey: ['diagnostics-list'],
    queryFn: listDiagnostics,
  });

  // Auto-select first file
  useEffect(() => {
    if (filesData?.files.length && !selectedFile) {
      setSelectedFile(filesData.files[0]?.filename ?? null);
    }
  }, [filesData, selectedFile]);

  // Fetch loss trajectory
  const { data: trajectoryData, isLoading: trajectoryLoading } = useQuery({
    queryKey: ['loss-trajectory', selectedFile, smoothingWindow],
    queryFn: () => getLossTrajectory(selectedFile!, smoothingWindow),
    enabled: !!selectedFile && activeTab === 'trajectories',
  });

  // Fetch fuzzy heatmap
  const { data: heatmapData, isLoading: heatmapLoading } = useQuery({
    queryKey: ['fuzzy-heatmap', selectedFile],
    queryFn: () => getFuzzyHeatmap(selectedFile!),
    enabled: !!selectedFile && activeTab === 'heatmap',
  });

  // Fetch epoch summary
  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: ['epoch-summary', selectedFile],
    queryFn: () => getEpochSummary(selectedFile!),
    enabled: !!selectedFile && (activeTab === 'summary' || activeTab === 'distributions'),
  });

  // Fetch gate distributions for selected epoch
  const { data: distributionData, isLoading: distributionLoading } = useQuery({
    queryKey: ['gate-distribution', selectedFile, selectedEpoch],
    queryFn: () => getGateDistribution(selectedFile!, selectedEpoch),
    enabled: !!selectedFile && activeTab === 'distributions',
  });

  // Update selected epoch when summary data arrives
  useEffect(() => {
    if (summaryData?.epochs.length) {
      const maxEpoch = Math.max(...summaryData.epochs.map((e) => e.epoch));
      setSelectedEpoch(maxEpoch);
    }
  }, [summaryData]);

  const tabs: { id: TabType; label: string; icon: string }[] = [
    { id: 'trajectories', label: 'Loss Trajectories', icon: '📈' },
    { id: 'heatmap', label: 'Fuzzy Heatmap', icon: '🔥' },
    { id: 'distributions', label: 'Gate Distributions', icon: '📊' },
    { id: 'summary', label: 'Epoch Summary', icon: '📋' },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-surface-100">Model Introspection</h1>
          <p className="mt-1 text-sm text-surface-400">
            Neural telemetry dashboard - watch CondorNet's brain form
          </p>
        </div>

        {/* File selector */}
        <div className="flex items-center gap-4">
          <div>
            <label className="input-label">Diagnostics File</label>
            <select
              className="input min-w-[200px]"
              value={selectedFile ?? ''}
              onChange={(e) => setSelectedFile(e.target.value)}
              disabled={filesLoading}
            >
              {filesLoading ? (
                <option>Loading...</option>
              ) : !filesData?.files.length ? (
                <option>No diagnostics files</option>
              ) : (
                filesData.files.map((file: DiagnosticsFile) => (
                  <option key={file.filename} value={file.filename}>
                    {file.filename} ({file.total_steps} steps)
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-2 border-b border-surface-800 pb-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 rounded-t-lg px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-brand-600/20 text-brand-400'
                : 'text-surface-400 hover:bg-surface-800 hover:text-surface-100'
            }`}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {!selectedFile ? (
        <div className="card flex h-96 items-center justify-center">
          <div className="text-center">
            <p className="text-lg text-surface-400">No diagnostics file selected</p>
            <p className="mt-2 text-sm text-surface-500">
              Run a training session with diagnostics enabled to see data here
            </p>
          </div>
        </div>
      ) : (
        <>
          {/* Loss Trajectories Tab */}
          {activeTab === 'trajectories' && (
            <div className="card">
              <div className="card-header flex items-center justify-between">
                <h2 className="text-lg font-semibold text-surface-100">
                  Loss Component Trajectories
                </h2>
                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 text-sm text-surface-400">
                    Smoothing:
                    <input
                      type="range"
                      min="10"
                      max="500"
                      step="10"
                      value={smoothingWindow}
                      onChange={(e) => setSmoothingWindow(Number(e.target.value))}
                      className="w-32"
                    />
                    <span className="w-12 text-surface-300">{smoothingWindow}</span>
                  </label>
                </div>
              </div>

              {trajectoryLoading ? (
                <LoadingSpinner fullScreen message="Loading trajectories..." />
              ) : trajectoryData ? (
                <LossTrajectoryChart
                  steps={trajectoryData.steps}
                  components={trajectoryData.components}
                  height={500}
                />
              ) : (
                <div className="flex h-64 items-center justify-center text-surface-500">
                  No trajectory data available
                </div>
              )}
            </div>
          )}

          {/* Fuzzy Heatmap Tab */}
          {activeTab === 'heatmap' && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-surface-100">
                  Fuzzy Gate Activation Heatmap
                </h2>
                <p className="mt-1 text-sm text-surface-500">
                  Watch which gates wake up, saturate, die, or oscillate over training
                </p>
              </div>

              {heatmapLoading ? (
                <LoadingSpinner fullScreen message="Loading heatmap..." />
              ) : heatmapData ? (
                <FuzzyGateHeatmap
                  steps={heatmapData.steps}
                  gates={heatmapData.gates}
                  values={heatmapData.values}
                  height={400}
                />
              ) : (
                <div className="flex h-64 items-center justify-center text-surface-500">
                  No fuzzy gate data available
                </div>
              )}
            </div>
          )}

          {/* Gate Distributions Tab */}
          {activeTab === 'distributions' && (
            <div className="card">
              <div className="card-header flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-surface-100">
                    Gate Activation Distributions
                  </h2>
                  <p className="mt-1 text-sm text-surface-500">
                    Analyze gate behavior: binary, soft, dead, or chaotic
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-surface-400">Epoch:</label>
                  <select
                    className="input w-24"
                    value={selectedEpoch}
                    onChange={(e) => setSelectedEpoch(Number(e.target.value))}
                  >
                    {summaryData?.epochs.map((e) => (
                      <option key={e.epoch} value={e.epoch}>
                        {e.epoch}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {distributionLoading ? (
                <LoadingSpinner fullScreen message="Loading distributions..." />
              ) : distributionData ? (
                <GateDistributionPanel
                  epoch={distributionData.epoch}
                  gates={distributionData.gates}
                />
              ) : (
                <div className="flex h-64 items-center justify-center text-surface-500">
                  No distribution data available
                </div>
              )}
            </div>
          )}

          {/* Epoch Summary Tab */}
          {activeTab === 'summary' && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-surface-100">
                  Per-Epoch Training Summary
                </h2>
                <p className="mt-1 text-sm text-surface-500">
                  Track MSE compression, pattern diversity, and metric evolution
                </p>
              </div>

              {summaryLoading ? (
                <LoadingSpinner fullScreen message="Loading summary..." />
              ) : summaryData ? (
                <EpochSummaryPanel epochs={summaryData.epochs} />
              ) : (
                <div className="flex h-64 items-center justify-center text-surface-500">
                  No epoch summary available
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Quick insights footer */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <InsightCard
          icon="🧠"
          title="What to Look For"
          items={[
            'MSE < 0.5 = good strike geometry',
            'Pattern entropy < -2 = diverse logic',
            'No dead gates = healthy network',
          ]}
        />
        <InsightCard
          icon="⚠️"
          title="Warning Signs"
          items={[
            'Sharpe stuck negative',
            'Gates saturating (all 1.0)',
            'Turnover spiking',
          ]}
        />
        <InsightCard
          icon="✅"
          title="Good Signs"
          items={[
            'MSE collapsing fast',
            'Binary gates forming',
            'Energy stabilizing',
          ]}
        />
        <InsightCard
          icon="🔧"
          title="Interventions"
          items={[
            'Increase pattern_ent weight',
            'Add gate dropout',
            'Adjust learning rate',
          ]}
        />
      </div>
    </div>
  );
}

function InsightCard({
  icon,
  title,
  items,
}: {
  icon: string;
  title: string;
  items: string[];
}) {
  return (
    <div className="rounded-lg border border-surface-800 bg-surface-900/50 p-4">
      <div className="flex items-center gap-2">
        <span className="text-xl">{icon}</span>
        <h3 className="font-medium text-surface-100">{title}</h3>
      </div>
      <ul className="mt-2 space-y-1">
        {items.map((item, i) => (
          <li key={i} className="text-xs text-surface-400">
            • {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
