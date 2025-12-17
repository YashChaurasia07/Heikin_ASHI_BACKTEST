import { useState, useEffect } from 'react';
import api from '../services/api';
import { Plus, Trash2, Upload, RefreshCw, Loader2, Database } from 'lucide-react';

interface Symbol {
  symbol: string;
  exchange: string;
  active: boolean;
  added_date?: string;
}

export default function SymbolManager() {
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [bulkSymbols, setBulkSymbols] = useState('');
  const [showBulkAdd, setShowBulkAdd] = useState(false);

  useEffect(() => {
    loadSymbols();
  }, []);

  const loadSymbols = async () => {
    try {
      setLoading(true);
      const data = await api.getAllSymbols();
      setSymbols(data);
    } catch (error) {
      console.error('Error loading symbols:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddSymbol = async () => {
    if (!newSymbol.trim()) return;
    
    try {
      await api.addSymbol(newSymbol.toUpperCase());
      setNewSymbol('');
      await loadSymbols();
    } catch (error) {
      console.error('Error adding symbol:', error);
      alert('Error adding symbol');
    }
  };

  const handleBulkAdd = async () => {
    const symbolList = bulkSymbols
      .split(/[\n,]/)
      .map(s => s.trim().toUpperCase())
      .filter(s => s.length > 0);
    
    if (symbolList.length === 0) return;

    try {
      const result = await api.bulkAddSymbols(symbolList);
      alert(`Added ${result.total_added} symbols, ${result.total_skipped} already existed`);
      setBulkSymbols('');
      setShowBulkAdd(false);
      await loadSymbols();
    } catch (error) {
      console.error('Error bulk adding symbols:', error);
      alert('Error adding symbols');
    }
  };

  const handleDeleteSymbol = async (symbol: string) => {
    if (!confirm(`Delete ${symbol}?`)) return;

    try {
      await api.deleteSymbol(symbol);
      await loadSymbols();
    } catch (error) {
      console.error('Error deleting symbol:', error);
      alert('Error deleting symbol');
    }
  };

  const handleSyncData = async () => {
    try {
      setSyncing(true);
      const result = await api.syncData();
      const successCount = result.filter((r: any) => r.status === 'success').length;
      alert(`Data sync completed: ${successCount}/${result.length} successful`);
    } catch (error) {
      console.error('Error syncing data:', error);
      alert('Error syncing data');
    } finally {
      setSyncing(false);
    }
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Database size={24} />
          Symbol Management
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setShowBulkAdd(!showBulkAdd)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 text-sm font-medium transition-colors"
          >
            <Upload size={16} />
            Bulk Add
          </button>
          <button
            onClick={handleSyncData}
            disabled={syncing}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg flex items-center gap-2 text-sm font-medium transition-colors"
          >
            {syncing ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
            Sync Data
          </button>
        </div>
      </div>

      {/* Add Single Symbol */}
      <div className="mb-4 flex gap-2">
        <input
          type="text"
          value={newSymbol}
          onChange={(e) => setNewSymbol(e.target.value)}
          placeholder="Symbol (e.g., RELIANCE)"
          className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500"
          onKeyPress={(e) => e.key === 'Enter' && handleAddSymbol()}
        />
        <button
          onClick={handleAddSymbol}
          className="px-6 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg flex items-center gap-2 font-medium transition-colors"
        >
          <Plus size={18} />
          Add
        </button>
      </div>

      {/* Bulk Add Section */}
      {showBulkAdd && (
        <div className="mb-4 p-4 bg-gray-700 border border-gray-600 rounded-lg">
          <h3 className="font-medium mb-2">Bulk Add Symbols</h3>
          <textarea
            value={bulkSymbols}
            onChange={(e) => setBulkSymbols(e.target.value)}
            placeholder="Enter symbols separated by commas or new lines&#10;Example: RELIANCE, TCS, INFY"
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 mb-2 focus:outline-none focus:border-blue-500"
            rows={5}
          />
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => setShowBulkAdd(false)}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg text-sm"
            >
              Cancel
            </button>
            <button
              onClick={handleBulkAdd}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium"
            >
              Add All
            </button>
          </div>
        </div>
      )}

      {/* Symbols List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {loading ? (
          <div className="text-center py-8 text-gray-400">
            <Loader2 size={32} className="animate-spin mx-auto" />
            <p className="mt-2">Loading symbols...</p>
          </div>
        ) : symbols.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            No symbols added yet
          </div>
        ) : (
          symbols.map((symbol) => (
            <div
              key={symbol.symbol}
              className="flex items-center justify-between bg-gray-700 px-4 py-3 rounded-lg hover:bg-gray-650 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="font-medium">{symbol.symbol}</span>
                <span className="text-sm text-gray-400">{symbol.exchange}</span>
              </div>
              <button
                onClick={() => handleDeleteSymbol(symbol.symbol)}
                className="text-red-500 hover:text-red-400 transition-colors"
              >
                <Trash2 size={18} />
              </button>
            </div>
          ))
        )}
      </div>

      <div className="mt-4 text-sm text-gray-400">
        Total: {symbols.length} symbols
      </div>
    </div>
  );
}
