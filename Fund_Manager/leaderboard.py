import json
import os

LEADERBOARD_FILE = os.path.join(os.path.dirname(__file__), "leaderboard.json")

class Leaderboard:
    def __init__(self):
        self.leaders = self._load()

    def _load(self):
        if os.path.exists(LEADERBOARD_FILE):
            try:
                with open(LEADERBOARD_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[Leaderboard] Warning: Corrupted leaderboard file, starting fresh: {e}")
                return []
            except IOError as e:
                print(f"[Leaderboard] Warning: Could not read leaderboard file: {e}")
                return []
        return []

    def _save(self):
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(self.leaders, f, indent=2)

    def update(self, name, metrics, code_snippet=""):
        """
        Updates the leaderboard if the new strategy is valid.
        metrics: dict containing 'Sharpe Ratio', 'Total Return', etc.
        """
        # Parse metrics if they are strings (heuristic)
        try:
            sharpe = float(metrics.get("Sharpe Ratio", 0) or 0)
            total_return = metrics.get("Total Return", 0)
            if isinstance(total_return, str):
                ret = float(total_return.replace("%", "").strip() or 0)
            else:
                ret = float(total_return or 0)
        except (ValueError, TypeError, AttributeError) as e:
            print(f"[Leaderboard] Warning: Could not parse metrics: {e}")
            sharpe = 0
            ret = 0

        entry = {
            "name": name,
            "sharpe": sharpe,
            "return": ret,
            "logic_summary": code_snippet[:500], # Store a snippet of the 'secret sauce'
            "timestamp": "now"
        }
        
        self.leaders.append(entry)
        # Sort by Sharpe Descending
        self.leaders.sort(key=lambda x: x["sharpe"], reverse=True)
        # Keep top 5
        self.leaders = self.leaders[:5]
        self._save()

    def get_champion(self):
        if self.leaders:
            return self.leaders[0]
        return None
