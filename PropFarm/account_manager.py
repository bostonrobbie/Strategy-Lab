import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class AccountManager:
    def __init__(self, data_file: str = "data/accounts.json"):
        self.data_file = data_file
        self.accounts = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Loads accounts from the JSON file."""
        if not os.path.exists(self.data_file):
            return []
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _save_data(self):
        """Saves current accounts list to JSON file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.accounts, f, indent=4)

    def add_account(self, account_data: Dict):
        """
        Adds a new account.
        account_data expected keys:
        - firm_name (str)
        - account_size (float)
        - account_type (str): 'Challenge', 'Verification', 'Funded'
        - status (str): 'Active', 'Passed', 'Failed', 'Payout Pending'
        - current_balance (float)
        - start_date (str)
        - daily_dd_limit_percent (float): e.g., 0.05 for 5%
        - max_dd_limit_percent (float): e.g., 0.10 for 10%
        - dd_type (str): 'Static', 'Trailing', 'Balance-Based'
        """
        # Generate a simple ID if not present
        if 'id' not in account_data:
            account_data['id'] = f"{account_data.get('firm_name', 'unk')}_{int(datetime.now().timestamp())}"
        
        # Ensure high water mark is initialized
        if 'high_water_mark' not in account_data:
            account_data['high_water_mark'] = account_data['current_balance']

        self.accounts.append(account_data)
        self._save_data()

    def update_account(self, account_id: str, updates: Dict):
        """Updates an existing account by ID."""
        for account in self.accounts:
            if account['id'] == account_id:
                # specific logic for High Water Mark updates if balance changes
                if 'current_balance' in updates:
                    new_bal = float(updates['current_balance'])
                    if new_bal > account.get('high_water_mark', 0):
                        account['high_water_mark'] = new_bal
                
                account.update(updates)
                self._save_data()
                return True
        return False

    def delete_account(self, account_id: str):
        """Removes an account by ID."""
        self.accounts = [acc for acc in self.accounts if acc['id'] != account_id]
        self._save_data()

    def get_all_accounts(self) -> List[Dict]:
        return self.accounts

    def get_account(self, account_id: str) -> Optional[Dict]:
        for acc in self.accounts:
            if acc['id'] == account_id:
                return acc
        return None

    def get_total_equity(self) -> float:
        return sum(float(acc.get('current_balance', 0)) for acc in self.accounts)

    def get_total_funded_capital(self) -> float:
        """Returns sum of account sizes for 'Funded' accounts only."""
        return sum(float(acc.get('account_size', 0)) for acc in self.accounts if acc.get('status') == 'Funded')
