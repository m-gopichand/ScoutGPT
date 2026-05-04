import json
from backend.src.tools.product import get_product_details
from backend.src.tools.search import search_products
res = get_product_details("B01LP0U5X0")
title = res.get("title", "")
print("Title:", title)
search_res = search_products(title, n=3)
print(json.dumps([p["asin"] for p in search_res.get("products", [])], indent=2))
