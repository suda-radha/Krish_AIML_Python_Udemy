# wap to find total cost of products in shopping cart




def find_total_cost(crt):
    total = 0;
    for item in crt:
        total += item["price"]*item["quantity"]
    return total

cart = [
    {"name": "Apple", "price": 0.3, "quantity": 10},
    {"name": "Mango", "price": 0.5, "quantity": 12},
    {"name": "Grapes", "price": 0.2, "quantity": 20}
]
print(find_total_cost(cart))
