import { useState, useEffect } from 'react';
import api from '../services/api';
import { Database, CheckCircle, AlertCircle, RefreshCw, Loader2 } from 'lucide-react';

interface BacktestStatus {
  total_symbols: number;
  daily: {
    ready: number;
    stale: number;
    ready_pct: number;
  };
  weekly: {
    ready: number;
    stale: number;
    ready_pct: number;
  };
}

export default function DataSyncStatus() {
  const [status, setStatus] = useState<BacktestStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      setLoading(true);
      const data = await api.getBacktestStatus();
      setStatus(data);
    } catch (error) {
      console.error('Error loading status:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSync = async () => {
    try {
      setSyncing(true);
      // Sync all data - the backend now syncs both daily and weekly automatically
      await api.syncData(undefined, 'daily', false);
      await loadStatus();
      alert('Data sync completed! Both daily and weekly data with Heikin Ashi have been synced.');
    } catch (error) {
      console.error('Error syncing:', error);
      alert('Error syncing data');
    } finally {
      setSyncing(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center py-8">
          <Loader2 size={32} className="animate-spin text-gray-400" />
        </div>
      </div>
    );
  }

  if (!status) return null;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Database size={24} />
          Data Sync Status
        </h2>
        <button
          onClick={handleSync}
          disabled={syncing}
          className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg flex items-center gap-2 text-sm font-medium transition-colors"
        >
          {syncing ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          Sync All
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Daily Data */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">Daily Data</h3>
            <span className="text-2xl font-bold text-green-400">
              {status.daily.ready_pct.toFixed(0)}%
            </span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400 flex items-center gap-2">
                <CheckCircle size={16} className="text-green-500" />
                Ready
              </span>
              <span className="font-medium">{status.daily.ready}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400 flex items-center gap-2">
                <AlertCircle size={16} className="text-yellow-500" />
                Stale
              </span>
              <span className="font-medium">{status.daily.stale}</span>
            </div>
          </div>
          <div className="mt-3 w-full bg-gray-600 rounded-full h-2">
            <div
              className="bg-green-500 h-2 rounded-full transition-all"
              style={{ width: `${status.daily.ready_pct}%` }}
            />
          </div>
        </div>

        {/* Weekly Data */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">Weekly Data</h3>
            <span className="text-2xl font-bold text-blue-400">
              {status.weekly.ready_pct.toFixed(0)}%
            </span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400 flex items-center gap-2">
                <CheckCircle size={16} className="text-green-500" />
                Ready
              </span>
              <span className="font-medium">{status.weekly.ready}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400 flex items-center gap-2">
                <AlertCircle size={16} className="text-yellow-500" />
                Stale
              </span>
              <span className="font-medium">{status.weekly.stale}</span>
            </div>
          </div>
          <div className="mt-3 w-full bg-gray-600 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all"
              style={{ width: `${status.weekly.ready_pct}%` }}
            />
          </div>
        </div>
      </div>

      <div className="mt-4 text-center text-sm text-gray-400">
        Total Symbols: {status.total_symbols}
      </div>
      
      <div className="mt-4 p-3 bg-blue-900/30 border border-blue-700 rounded-lg">
        <p className="text-xs text-blue-300">
          ℹ️ <strong>Smart Sync:</strong> Only fetches missing data since last sync. 
          Processes up to 500 stocks in parallel. Syncs both daily & weekly intervals with Heikin Ashi.
        </p>
      </div>
      
      <div className="mt-3 p-3 bg-yellow-900/30 border border-yellow-700 rounded-lg">
        <p className="text-xs text-yellow-300">
          ⚠️ <strong>Note:</strong> Backtest does NOT sync data automatically. 
          Please sync data manually before running backtest.
        </p>
      </div>
    </div>
  );
}
