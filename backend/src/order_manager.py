# order_manager.py
import json
import logging

logger = logging.getLogger("order_manager")

def save_order_to_json(drink_type, size, milk, extras, name):
    """
    Saves the order details to a JSON file.
    """
    order_data = {
        "drinkType": drink_type,
        "size": size,
        "milk": milk,
        "extras": extras,
        "name": name
    }
    
    file_name = "order_summary.json"
    
    try:
        with open(file_name, "w") as f:
            json.dump(order_data, f, indent=2)
        
        logger.info(f"Order successfully saved to {file_name}: {order_data}")
        return "Order saved successfully."
    except Exception as e:
        logger.error(f"Failed to save order: {e}")
        return "Failed to save order."