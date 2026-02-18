from datetime import datetime, timedelta
from typing import Dict, Optional

class PayoutTracker:
    @staticmethod
    def get_next_payout_date(account: Dict) -> Optional[str]:
        """
        Calculates the next eligible payout date based on account status and start date.
        Returns ISO format date string YYYY-MM-DD or None.
        """
        status = account.get('status', 'Challenge')
        if status != 'Funded':
            return None
            
        # Logic: 
        # Standard: 14 days after first trade/start date for first payout.
        # Then every 14 days.
        # This is a simplification. Real firms have specific cycles.
        
        start_date_str = account.get('start_date')
        if not start_date_str:
            return None
            
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            return None
            
        # Simple Logic: Next payout is 14 days from start, or if that's passed, 
        # the next 14-day interval from today.
        
        first_payout = start_date + timedelta(days=14)
        today = datetime.now()
        
        if first_payout > today:
            return first_payout.strftime("%Y-%m-%d")
        else:
            # If passed, find next cycle
            days_since_start = (today - start_date).days
            cycles = (days_since_start // 14) + 1
            next_payout = start_date + timedelta(days=cycles * 14)
            return next_payout.strftime("%Y-%m-%d")

    @staticmethod
    def get_days_until_payout(account: Dict) -> int:
        target_date_str = PayoutTracker.get_next_payout_date(account)
        if not target_date_str:
            return -1
            
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        delta = target_date - datetime.now()
        return delta.days
