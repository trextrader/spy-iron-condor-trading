/**
 * Training Telemetry Hook
 * Phase 6.7 - Model Introspection
 *
 * Subscribes to real-time training updates via WebSocket.
 */

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';

export interface TrainingStep {
  step: number;
  epoch: number;
  loss: number;
  mse: number;
  npdd: number;
  sharpe: number;
  dd: number;
  turnover: number;
  fuzzy: number;
  pattern_ent: number;
  group_inv: number;
  rho: number;
  energy: number;
  growth: number;
  lr?: number;
  grad_norm?: number;
  scaler_scale?: number;
  timestamp?: string;
}

export interface TrainingStatus {
  isTraining: boolean;
  currentEpoch: number;
  currentStep: number;
  totalSteps: number;
  progressPct: number;
  etaSeconds?: number;
  startedAt?: string;
}

export interface FuzzyActivations {
  step: number;
  epoch: number;
  activations: number[];
}

interface UseTrainingTelemetryReturn {
  // Current state
  status: TrainingStatus;
  latestStep: TrainingStep | null;
  latestFuzzy: FuzzyActivations | null;

  // Historical data (last N steps)
  stepHistory: TrainingStep[];
  fuzzyHistory: FuzzyActivations[];

  // Connection
  isConnected: boolean;

  // Actions
  clearHistory: () => void;
}

const MAX_HISTORY = 1000; // Keep last 1000 steps in memory

export function useTrainingTelemetry(): UseTrainingTelemetryReturn {
  const { isConnected, subscribe } = useWebSocket();

  const [status, setStatus] = useState<TrainingStatus>({
    isTraining: false,
    currentEpoch: 0,
    currentStep: 0,
    totalSteps: 0,
    progressPct: 0,
  });

  const [latestStep, setLatestStep] = useState<TrainingStep | null>(null);
  const [latestFuzzy, setLatestFuzzy] = useState<FuzzyActivations | null>(null);
  const [stepHistory, setStepHistory] = useState<TrainingStep[]>([]);
  const [fuzzyHistory, setFuzzyHistory] = useState<FuzzyActivations[]>([]);

  // Handle training step updates
  const handleTrainingStep = useCallback((data: unknown) => {
    const step = data as TrainingStep;
    setLatestStep(step);

    // Update status
    setStatus((prev) => ({
      ...prev,
      isTraining: true,
      currentEpoch: step.epoch,
      currentStep: step.step,
    }));

    // Append to history (with limit)
    setStepHistory((prev) => {
      const next = [...prev, step];
      if (next.length > MAX_HISTORY) {
        return next.slice(-MAX_HISTORY);
      }
      return next;
    });
  }, []);

  // Handle fuzzy activation updates
  const handleFuzzyUpdate = useCallback((data: unknown) => {
    const fuzzy = data as FuzzyActivations;
    setLatestFuzzy(fuzzy);

    // Append to history (with limit)
    setFuzzyHistory((prev) => {
      const next = [...prev, fuzzy];
      if (next.length > MAX_HISTORY) {
        return next.slice(-MAX_HISTORY);
      }
      return next;
    });
  }, []);

  // Handle training status updates
  const handleStatusUpdate = useCallback((data: unknown) => {
    const statusData = data as Partial<TrainingStatus>;
    setStatus((prev) => ({
      ...prev,
      ...statusData,
    }));
  }, []);

  // Handle training complete
  const handleTrainingComplete = useCallback((data: unknown) => {
    setStatus((prev) => ({
      ...prev,
      isTraining: false,
      progressPct: 100,
    }));
  }, []);

  // Subscribe to channels
  useEffect(() => {
    if (!isConnected) return;

    const unsubStep = subscribe('training.step', handleTrainingStep);
    const unsubFuzzy = subscribe('training.fuzzy', handleFuzzyUpdate);
    const unsubStatus = subscribe('training.status', handleStatusUpdate);
    const unsubComplete = subscribe('training.complete', handleTrainingComplete);

    return () => {
      unsubStep();
      unsubFuzzy();
      unsubStatus();
      unsubComplete();
    };
  }, [
    isConnected,
    subscribe,
    handleTrainingStep,
    handleFuzzyUpdate,
    handleStatusUpdate,
    handleTrainingComplete,
  ]);

  // Clear history
  const clearHistory = useCallback(() => {
    setStepHistory([]);
    setFuzzyHistory([]);
    setLatestStep(null);
    setLatestFuzzy(null);
  }, []);

  return {
    status,
    latestStep,
    latestFuzzy,
    stepHistory,
    fuzzyHistory,
    isConnected,
    clearHistory,
  };
}

// Store for training telemetry (can be used outside React)
import { create } from 'zustand';

interface TrainingTelemetryStore {
  status: TrainingStatus;
  latestStep: TrainingStep | null;
  stepHistory: TrainingStep[];

  setStatus: (status: Partial<TrainingStatus>) => void;
  addStep: (step: TrainingStep) => void;
  clearHistory: () => void;
}

export const useTrainingStore = create<TrainingTelemetryStore>((set) => ({
  status: {
    isTraining: false,
    currentEpoch: 0,
    currentStep: 0,
    totalSteps: 0,
    progressPct: 0,
  },
  latestStep: null,
  stepHistory: [],

  setStatus: (statusUpdate) =>
    set((state) => ({
      status: { ...state.status, ...statusUpdate },
    })),

  addStep: (step) =>
    set((state) => ({
      latestStep: step,
      stepHistory: [...state.stepHistory, step].slice(-MAX_HISTORY),
      status: {
        ...state.status,
        isTraining: true,
        currentEpoch: step.epoch,
        currentStep: step.step,
      },
    })),

  clearHistory: () =>
    set({
      latestStep: null,
      stepHistory: [],
    }),
}));
