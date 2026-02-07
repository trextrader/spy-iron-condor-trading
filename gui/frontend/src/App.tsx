import { Routes, Route, Navigate } from 'react-router-dom';
import { Suspense, lazy } from 'react';
import Layout from '@/components/layout/Layout';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { WebSocketProvider } from '@/providers/WebSocketProvider';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const BacktestPanel = lazy(() => import('@/pages/BacktestPanel'));
const IntelligencePanel = lazy(() => import('@/pages/IntelligencePanel'));
const ExecutionPanel = lazy(() => import('@/pages/ExecutionPanel'));
const TapeViewer = lazy(() => import('@/pages/TapeViewer'));
const ModelIntrospection = lazy(() => import('@/pages/ModelIntrospection'));
const TradeExplorer = lazy(() => import('@/pages/TradeExplorer'));
const OptimizationPanel = lazy(() => import('@/pages/OptimizationPanel'));

function App() {
  return (
    <WebSocketProvider>
      <Layout>
        <Suspense fallback={<LoadingSpinner fullScreen />}>
          <Routes>
            {/* Dashboard - Home */}
            <Route path="/" element={<Dashboard />} />

            {/* Control Panels */}
            <Route path="/backtest" element={<BacktestPanel />} />
            <Route path="/intelligence" element={<IntelligencePanel />} />
            <Route path="/execution" element={<ExecutionPanel />} />

            {/* Data Views */}
            <Route path="/tape" element={<TapeViewer />} />
            <Route path="/model" element={<ModelIntrospection />} />
            <Route path="/trades" element={<TradeExplorer />} />

            {/* Advanced */}
            <Route path="/optimization" element={<OptimizationPanel />} />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </Layout>
    </WebSocketProvider>
  );
}

export default App;
