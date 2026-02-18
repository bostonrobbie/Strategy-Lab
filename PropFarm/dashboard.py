import streamlit as st
import pandas as pd
from datetime import datetime
from account_manager import AccountManager
from risk_engine import RiskEngine
from payouts import PayoutTracker

# Page Config
st.set_page_config(layout="wide", page_title="PropFarm Command Center", page_icon="🚜")

# Initialize Manager
manager = AccountManager()

# --- Sidebar: Account Actions ---
st.sidebar.header("🚜 PropFarm Manager")

with st.sidebar.expander("➕ Add New Account"):
    with st.form("add_account_form"):
        firm = st.text_input("Firm Name (e.g. Apex, FTMO)")
        size = st.number_input("Account Size ($)", min_value=5000, step=5000, value=50000)
        atype = st.selectbox("Type", ["Challenge", "Verification", "Funded"])
        dd_type = st.selectbox("Drawdown Type", ["Balance-Based", "Trailing", "Static"])
        submit_acc = st.form_submit_button("Add Account")
        
        if submit_acc:
            new_acc = {
                "firm_name": firm,
                "account_size": size,
                "status": atype,  # Initial status matches type usually
                "account_type": atype,
                "current_balance": size,
                "bod_balance": size, # Initialize BOD balance as current
                "start_date": datetime.now().strftime("%Y-%m-%d"),
                "daily_dd_limit_percent": 0.05, # Default, user can edit later
                "max_dd_limit_percent": 0.10,
                "dd_type": dd_type
            }
            manager.add_account(new_acc)
            st.success(f"Added {firm} account!")
            st.rerun()

# --- Main Dashboard ---

st.title("PropFarm Command Center")

# Metrics Row
accounts = manager.get_all_accounts()
total_equity = manager.get_total_equity()
funded_cap = manager.get_total_funded_capital()
total_accts = len(accounts)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Equity", f"${total_equity:,.2f}")
col2.metric("Funded Capital", f"${funded_cap:,.0f}")
col3.metric("Active Accounts", total_accts)
col4.metric("Est. Monthly Payout (3%)", f"${funded_cap * 0.03:,.2f}")

st.markdown("---")

# Account Grid
st.subheader("🏁 Active Accounts & Risk Status")

if not accounts:
    st.info("No accounts tracked. Add one in the sidebar!")
else:
    for acc in accounts:
        # Calculate Risk
        risk_data = RiskEngine.calculate_daily_risk(acc)
        payout_days = PayoutTracker.get_days_until_payout(acc)
        
        # Color coding
        status_color = "green"
        if risk_data['true_remaining_risk'] < 0:
            status_color = "red"
        elif risk_data['true_remaining_risk'] < 1000:
            status_color = "orange"
        
        with st.container(border=True):
            cols = st.columns([2, 2, 2, 2, 1])
            
            # Col 1: Name & ID
            with cols[0]:
                st.markdown(f"**{acc['firm_name']}** ({acc['account_type']})")
                st.caption(f"ID: {acc['id']}")
                if payout_days is not None and payout_days >= 0:
                    st.caption(f"💰 Payout: {payout_days} days")
            
            # Col 2: Balance Input
            with cols[1]:
                new_bal = st.number_input("Current Balance", value=float(acc.get('current_balance', 0)), key=f"bal_{acc['id']}")
                if new_bal != float(acc.get('current_balance', 0)):
                    manager.update_account(acc['id'], {"current_balance": new_bal})
                    st.rerun()

            # Col 3: Risk Metrics
            with cols[2]:
                st.markdown(f"**Daily Limit**: ${risk_data['daily_breach_level']:,.2f}")
                st.markdown(f"**Total Limit**: ${risk_data['total_breach_level']:,.2f}")
            
            # Col 4: Buffer
            with cols[3]:
                st.markdown(f"**Remaining Risk**: :{('red' if risk_data['true_remaining_risk'] < 0 else status_color)}[${risk_data['true_remaining_risk']:,.2f}]")
                st.markdown(f"**Safe Risk (Trade)**: ${risk_data['safe_remaining_risk']:,.2f}")
            
            # Col 5: Actions
            with cols[4]:
                if st.button("Config", key=f"cfg_{acc['id']}"):
                    st.toast("Configuration View Coming Soon")
                if st.button("🗑️", key=f"del_{acc['id']}"):
                    manager.delete_account(acc['id'])
                    st.rerun()
            
            # Progress Bar for Daily DD
            # Calculate % used
            bod = float(acc.get('bod_balance', acc['account_size']))
            daily_limit = bod * float(acc.get('daily_dd_limit_percent', 0.05))
            current_loss = bod - float(acc.get('current_balance', bod))
            if current_loss < 0: current_loss = 0 # Profit
            
            pct_used = min(current_loss / daily_limit, 1.0) if daily_limit > 0 else 0
            st.progress(pct_used, text=f"Daily Drawdown Used: {pct_used*100:.1f}%")

st.markdown("---")
st.subheader("💡 Strategy & Operations")
st.markdown("""
*   **Operational**: Check your 'Payouts' caption on each funded account.
*   **Tactical**: If 'Remaining Risk' is > 2% of account, consider 'Aggressive' mode. If < 1%, switch to 'Conservative'.
""")
