import sqlite3
import json
import os
import hashlib
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

class StrategyRegistry:
    """
    Enhanced Trading Knowledge Base.
    Persists metrics, source code, notes, and market regime context.
    Provides duplicate detection to prevent redundant research.
    """
    def __init__(self, db_path: str = "backtests.db"):
        self.db_path = db_path
        self._init_db()
        self._migrate_db()

    def _init_db(self):
        """Initialize SQLite database schema with extended Knowledge Base fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for backtest runs with KB extensions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT,
                symbol TEXT,
                interval TEXT,
                params_json TEXT,
                total_return REAL,
                cagr REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                calmar_ratio REAL,
                profit_factor REAL,
                var_95 REAL,
                ending_equity REAL,
                data_range_start TEXT,
                data_range_end TEXT,
                regime TEXT,
                notes TEXT,
                source_code TEXT,
                hash_id TEXT UNIQUE
            )
        """)

        # Table for System Health Logs (QA Specs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT, -- PASS, WARNING, FAIL
                report_json TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Adds missing columns to existing database if needed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing columns
        cursor.execute("PRAGMA table_info(backtest_runs)")
        columns = [row[1] for row in cursor.fetchall()]
        
        new_cols = {
            'regime': 'TEXT',
            'notes': 'TEXT',
            'source_code': 'TEXT',
            'hash_id': 'TEXT'
        }
        
        for col, col_type in new_cols.items():
            if col not in columns:
                print(f"Migrating database: Adding column {col}...")
                try:
                    cursor.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {col_type}")
                    if col == 'hash_id':
                        print("Creating unique index for hash_id...")
                        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_hash_id ON backtest_runs(hash_id)")
                except Exception as e:
                    print(f"Error adding column {col}: {e}")
        
        conn.commit()
        conn.close()

    def _generate_hash(self, strategy_name: str, symbol: str, interval: str, params: Dict[str, Any], data_range: tuple) -> str:
        """Generates a unique hash for a backtest configuration."""
        hash_str = f"{strategy_name}_{symbol}_{interval}_{json.dumps(params, sort_keys=True)}_{data_range}"
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def check_duplicate(self, strategy_name: str, symbol: str, interval: str, params: Dict[str, Any], data_range: tuple) -> Optional[Dict[str, Any]]:
        """Checks if a backtest with identical config already exists."""
        hash_id = self._generate_hash(strategy_name, symbol, interval, params, data_range)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, sharpe_ratio, total_return FROM backtest_runs WHERE hash_id = ?", (hash_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'timestamp': row[0],
                'sharpe_ratio': row[1],
                'total_return': row[2]
            }
        return None

    def save_run(self, strategy_name: str, symbol: str, interval: str, params: Dict[str, Any], stats: Dict[str, Any], 
                 data_range: tuple, regime: str = "UNKNOWN", notes: str = "", source_code: str = ""):
        """Persist a backtest run with full context to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date, end_date = data_range
        hash_id = self._generate_hash(strategy_name, symbol, interval, params, data_range)
        
        try:
            cursor.execute("""
                INSERT INTO backtest_runs (
                    strategy_name, symbol, interval, params_json, 
                    total_return, cagr, sharpe_ratio, max_drawdown, 
                    calmar_ratio, profit_factor, var_95, ending_equity,
                    data_range_start, data_range_end, regime, notes, source_code, hash_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name,
                symbol,
                interval,
                json.dumps(params),
                stats.get('Total Return', 0.0),
                stats.get('CAGR', 0.0),
                stats.get('Sharpe Ratio', 0.0),
                stats.get('Max Drawdown', 0.0),
                stats.get('Calmar Ratio', 0.0),
                stats.get('Profit Factor', 0.0),
                stats.get('VaR (95%)', 0.0),
                stats.get('Ending Equity', 0.0),
                str(start_date),
                str(end_date),
                regime,
                notes,
                source_code,
                hash_id
            ))
            conn.commit()
            print(f"Knowledge Base Updated: Run archived with notes: '{notes[:30]}...'")
        except sqlite3.IntegrityError:
            print("Warning: Identity run already exists. Registry entry not duplicated.")
        finally:
            conn.close()

    def save_batch(self, runs: List[Dict[str, Any]]):
        """
        Batch insert multiple runs.
        Expects list of dicts with keys matching save_run arguments.
        """
        if not runs:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_to_insert = []
        
        for run in runs:
            # Extract fields
            params = run['params']
            stats = run['stats']
            data_range = run['data_range']
            
            hash_id = self._generate_hash(
                run['strategy_name'], 
                run['symbol'], 
                run['interval'], 
                params, 
                data_range
            )
            
            data_to_insert.append((
                run['strategy_name'],
                run['symbol'],
                run['interval'],
                json.dumps(params),
                stats.get('Total Return', 0.0),
                stats.get('CAGR', 0.0),
                stats.get('Sharpe Ratio', 0.0),
                stats.get('Max Drawdown', 0.0),
                stats.get('Calmar Ratio', 0.0),
                stats.get('Profit Factor', 0.0),
                stats.get('VaR (95%)', 0.0),
                stats.get('Ending Equity', 0.0),
                str(data_range[0]),
                str(data_range[1]),
                run.get('regime', "UNKNOWN"),
                run.get('notes', ""),
                run.get('source_code', ""),
                hash_id
            ))

        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO backtest_runs (
                    strategy_name, symbol, interval, params_json, 
                    total_return, cagr, sharpe_ratio, max_drawdown, 
                    calmar_ratio, profit_factor, var_95, ending_equity,
                    data_range_start, data_range_end, regime, notes, source_code, hash_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            conn.commit()
            print(f"[REGISTRY] Batch archived {cursor.rowcount} runs to Knowledge Base.")
        except Exception as e:
            print(f"[REGISTRY] Batch Save Error: {e}")
        finally:
            conn.close()

    def get_leaderboard(self, limit: int = 10) -> pd.DataFrame:
        """Retrieve top historical results with notes and regimes."""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT strategy_name, symbol, interval, sharpe_ratio, total_return, regime, notes, timestamp
            FROM backtest_runs
            ORDER BY sharpe_ratio DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df

    def get_historical_average(self, strategy_name: str, symbol: Optional[str] = None) -> Dict[str, float]:
        """Calculate average metrics for a strategy/symbol combination."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT AVG(sharpe_ratio), AVG(total_return), AVG(max_drawdown) FROM backtest_runs WHERE strategy_name = ?"
        params = [strategy_name]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] is not None:
            return {
                'avg_sharpe': row[0],
                'avg_return': row[1],
                'avg_drawdown': row[2]
            }
        return {}

    def log_diagnostic(self, status: str, report: Dict[str, Any]):
        """Persist a system health check report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO system_health (status, report_json) VALUES (?, ?)", 
                           (status, json.dumps(report)))
            conn.commit()
            print("[MEMORY] Diagnostic Report Archived to Database.")
        except Exception as e:
            print(f"[MEMORY] Failed to archive diagnostic: {e}")
        finally:
            conn.close()
