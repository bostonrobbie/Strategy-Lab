from typing import Dict, Tuple

class RiskEngine:
    @staticmethod
    def calculate_daily_risk(account: Dict) -> Dict:
        """
        Calculates the maximum loss allowed for the day and remaining buffer.
        
        Returns:
            Dict containing:
            - 'daily_loss_limit_amount': The total $ amount allowed to be lost in a day
            - 'hard_breach_level': The price level where the account is breached
            - 'remaining_daily_risk': How much more can be lost today before breach
            - 'max_position_risk': Suggested max risk for next trade (e.g. 1/3 of remaining daily risk)
        """
        firm_name = account.get('firm_name', 'Generic')
        balance = float(account.get('current_balance', 0.0))
        size = float(account.get('account_size', 0.0))
        dd_percent = float(account.get('daily_dd_limit_percent', 0.05)) # Default 5%
        dd_type = account.get('dd_type', 'Balance-Based')
        
        # 1. Calculate Daily Breach Level
        # Most firms: Daily Loss is calculated based on the Starting Equity of the day OR Balance.
        # Since we don't have "Start of Day" balance automatically without a database of history,
        # we will assume the User inputs "Start of Day Balance" or we use a simplified logic 
        # where we assume the limit is purely static from the account size for now, 
        # BUT accurate tracking requires knowing the BOD (Beginning of Day) balance.
        
        # TODO: In a real app, we need to store 'bod_balance' every day at 00:00 UTC.
        # For this version, we will require the user to ensure 'bod_balance' is updated 
        # or we default it to current balance if missing (which is safer/conservative logic).
        
        bod_balance = float(account.get('bod_balance', balance))
        
        daily_loss_limit_amount = bod_balance * dd_percent
        daily_breach_level = bod_balance - daily_loss_limit_amount
        
        # 2. Calculate Max Breach Level (Total DD)
        max_dd_percent = float(account.get('max_dd_limit_percent', 0.10))
        
        if dd_type == 'Trailing':
            # Trailing from High Water Mark (HWM)
            hwm = float(account.get('high_water_mark', size))
            total_breach_level = hwm * (1 - max_dd_percent)
            # Some firms lock the trailing DD at the starting balance (e.g. Apex)
            # If HWM - max_dd > Starting Balance, usually it locks at Starting Balance.
            if total_breach_level > size:
                total_breach_level = size 
        else:
            # Static from Inital Size
            total_breach_level = size * (1 - max_dd_percent)

        # 3. Determine actual remaining buffer
        # The account is breached if specific Daily OR Total level is hit.
        
        dist_to_daily_fail = balance - daily_breach_level
        dist_to_total_fail = balance - total_breach_level
        
        # The actual "Stop Trading" level is whichever is closer
        true_remaining_risk = min(dist_to_daily_fail, dist_to_total_fail)
        
        # Safety buffer (don't get stopped out exactly at the line)
        safety_buffer = true_remaining_risk * 0.05 # 5% buffer
        safe_remaining_risk = true_remaining_risk - safety_buffer
        
        if safe_remaining_risk < 0:
            safe_remaining_risk = 0

        return {
            "daily_breach_level": daily_breach_level,
            "total_breach_level": total_breach_level,
            "true_remaining_risk": true_remaining_risk,
            "safe_remaining_risk": safe_remaining_risk,
            "status": "Safe" if true_remaining_risk > 0 else "BREACHED"
        }
