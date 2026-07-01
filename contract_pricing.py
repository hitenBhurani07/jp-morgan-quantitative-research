from datetime import datetime
import math
# Assuming nat_gas_pricing.py is in the same directory and contains estimate_price
from nat_gas_pricing import estimate_price

def price_storage_contract(
    injection_dates: list[str],
    withdrawal_dates: list[str],
    injection_volumes: list[float],
    withdrawal_volumes: list[float],
    max_storage_volume: float,
    max_injection_rate: float,
    max_withdrawal_rate: float,
    storage_cost_per_month: float,
    injection_cost_per_mmbtu: float,
    withdrawal_cost_per_mmbtu: float,
    transport_cost_per_transaction: float
) -> float:
    """
    Prices a natural gas storage contract based on expected cash flows.
    
    Args:
        injection_dates (list[str]): List of dates when gas is injected (bought).
        withdrawal_dates (list[str]): List of dates when gas is withdrawn (sold).
        injection_volumes (list[float]): Volume of gas injected on each date (in MMBtu).
        withdrawal_volumes (list[float]): Volume of gas withdrawn on each date (in MMBtu).
        max_storage_volume (float): Maximum volume that can be stored at any time.
        max_injection_rate (float): Max volume that can be injected per transaction.
        max_withdrawal_rate (float): Max volume that can be withdrawn per transaction.
        storage_cost_per_month (float): Fixed storage fee paid per month.
        injection_cost_per_mmbtu (float): Cost to inject 1 MMBtu.
        withdrawal_cost_per_mmbtu (float): Cost to withdraw 1 MMBtu.
        transport_cost_per_transaction (float): Fixed cost for each transport (injection or withdrawal).
        
    Returns:
        float: The net value of the contract.
    """
    
    # 1. Validation Checks
    if len(injection_dates) != len(injection_volumes):
        raise ValueError("Mismatch between injection dates and volumes.")
    if len(withdrawal_dates) != len(withdrawal_volumes):
        raise ValueError("Mismatch between withdrawal dates and volumes.")
        
    for vol in injection_volumes:
        if vol > max_injection_rate:
            raise ValueError(f"Injection volume {vol} exceeds max injection rate {max_injection_rate}.")
            
    for vol in withdrawal_volumes:
        if vol > max_withdrawal_rate:
            raise ValueError(f"Withdrawal volume {vol} exceeds max withdrawal rate {max_withdrawal_rate}.")
            
    # Combine all events and sort chronologically to track storage limits
    events = []
    for d, v in zip(injection_dates, injection_volumes):
        events.append({'date': datetime.strptime(d, '%Y-%m-%d'), 'type': 'inject', 'volume': v})
    for d, v in zip(withdrawal_dates, withdrawal_volumes):
        events.append({'date': datetime.strptime(d, '%Y-%m-%d'), 'type': 'withdraw', 'volume': v})
        
    events.sort(key=lambda x: x['date'])
    
    current_inventory = 0.0
    for event in events:
        if event['type'] == 'inject':
            current_inventory += event['volume']
            if current_inventory > max_storage_volume:
                raise ValueError(f"Storage exceeded max capacity on {event['date'].strftime('%Y-%m-%d')}.")
        elif event['type'] == 'withdraw':
            current_inventory -= event['volume']
            if current_inventory < 0:
                raise ValueError(f"Attempted to withdraw more than available inventory on {event['date'].strftime('%Y-%m-%d')}.")
    
    if current_inventory != 0:
        print("Warning: Contract ends with non-zero inventory in storage.")

    # 2. Calculate Cash Flows
    total_cost = 0.0
    total_revenue = 0.0
    
    # Injections (Cash Outflow)
    for date_str, vol in zip(injection_dates, injection_volumes):
        price_per_mmbtu = estimate_price(date_str)
        cost_of_gas = price_per_mmbtu * vol
        injection_cost = injection_cost_per_mmbtu * vol
        total_cost += cost_of_gas + injection_cost + transport_cost_per_transaction
        
    # Withdrawals (Cash Inflow)
    for date_str, vol in zip(withdrawal_dates, withdrawal_volumes):
        price_per_mmbtu = estimate_price(date_str)
        revenue_from_gas = price_per_mmbtu * vol
        withdrawal_cost = withdrawal_cost_per_mmbtu * vol
        
        total_revenue += revenue_from_gas
        total_cost += withdrawal_cost + transport_cost_per_transaction
        
    # 3. Storage Costs (Time-based)
    if events:
        start_date = events[0]['date']
        end_date = events[-1]['date']
        months_stored = math.ceil((end_date - start_date).days / 30.0)
        # Assuming you pay for the number of months the contract spans
        total_storage_cost = months_stored * storage_cost_per_month
        total_cost += total_storage_cost

    # 4. Final Value
    contract_value = total_revenue - total_cost
    
    return round(contract_value, 2)

if __name__ == "__main__":
    from nat_gas_pricing import load_and_fit_model
    
    # Ensure the model is loaded first so estimate_price works
    print("Loading data and fitting model...")
    load_and_fit_model('Nat_Gas.csv')
    print("-" * 50)
    
    # Example Scenario Based on Prompt:
    # "purchase a million MMBtu in summer at $2/MMBtu (simulated by our model), 
    # store this for four months, sell at $3/MMBtu (simulated).
    # $100K a month fixed fee, $10K per 1M MMBtu inject/withdraw, $50K transport per transaction"
    
    test_injection_dates = ['2023-06-15']
    test_withdrawal_dates = ['2023-10-15']
    test_volumes = [1_000_000] # 1 million MMBtu
    
    try:
        value = price_storage_contract(
            injection_dates=test_injection_dates,
            withdrawal_dates=test_withdrawal_dates,
            injection_volumes=test_volumes,
            withdrawal_volumes=test_volumes,
            max_storage_volume=2_000_000,
            max_injection_rate=2_000_000,
            max_withdrawal_rate=2_000_000,
            storage_cost_per_month=100_000,
            injection_cost_per_mmbtu=0.01, # $10K per 1M = $0.01 per MMBtu
            withdrawal_cost_per_mmbtu=0.01,
            transport_cost_per_transaction=50_000
        )
        print(f"Calculated Contract Value: ${value:,.2f}")
    except Exception as e:
        print(f"Error pricing contract: {e}")
