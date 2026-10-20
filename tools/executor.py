"""
Real (mocked) implementations of the 34 tools in data/tool_pool.json.

Model-agnostic on purpose: every one of the three model pipelines
(voxMind offline, step-audio2 offline, VoiceChat offline/native) is meant to
call the SAME `MockToolExecutor` for the SAME tool with the SAME args and get
the SAME result, so a difference in scores reflects a difference in model
behavior, not a difference in what each pipeline's tool stub happened to
return.

Two kinds of tools, matching data/tool_registry.json's `suggested_db_table`:
  - DB-backed (17 tools): read/write against a JSON table under tools/db/.
    A lookup miss deterministically synthesizes a row (stable hash of the
    query args, not random) and upserts it, so a query never hard-fails and
    repeated calls with the same args always return the same result.
  - Pure/computed (13 tools, no suggested_db_table): deterministic function
    of the args only, same spirit as Full-Duplex-Bench-v3's mock_apis.py.

Usage:
    executor = MockToolExecutor(allowed_tools=["search_flights"])  # one instance per evaluated example;
                                            # allowed_tools = the record's available_tools
    result = executor.call("search_flights", destination="Denver", date="next Friday", origin="Boston")
    executor.tool_results()                 # -> [{"name", "args", "result"}, ...] in call order
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "db"


def _stable_int(*parts: Any, lo: int, hi: int) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return lo + int(h, 16) % (hi - lo + 1)


def _stable_choice(options: List[Any], *parts: Any) -> Any:
    return options[_stable_int(*parts, lo=0, hi=len(options) - 1)]


class MockToolExecutor:
    def __init__(self, allowed_tools: Optional[Iterable[str]] = None):
        self.db: Dict[str, Any] = {}
        for f in sorted(DB_DIR.glob("*.json")):
            self.db[f.name] = json.loads(f.read_text())
        self.calls: List[dict] = []  # [{"name", "args", "result"}], call order preserved
        # The record's available_tools. None = every tool in TOOLS (old behavior).
        # A tool that exists in TOOLS but was not offered for this record must not
        # execute -- otherwise a model that invents a name which happens to collide
        # with another record's tool gets real data back and can "solve" a
        # cannot_answer request.
        self.allowed_tools = None if allowed_tools is None else set(allowed_tools)

    def call(self, name: str, **kwargs) -> dict:
        fn = self.TOOLS.get(name)
        if self.allowed_tools is not None and name not in self.allowed_tools:
            fn = None
        if fn is None:
            result = {"status": "error", "message": f"Unknown tool: {name}"}
        else:
            try:
                result = fn(self, **kwargs)
            except TypeError as e:
                result = {"status": "error", "message": f"Bad arguments for {name}({kwargs}): {e}"}
        self.calls.append({"name": name, "args": kwargs, "result": result})
        return result

    def tool_results(self) -> List[dict]:
        """Full call log for this session, in the shape the evaluator expects."""
        return deepcopy(self.calls)

    # ================= fdb_v3 / "core" (DB-backed) =================

    def add_to_cart(self, product_id, quantity=1, **kw):
        cart = self.db["cart.json"]
        cart["items"].append({"product_id": product_id, "quantity": quantity})
        cart["total"] = round(cart["total"] + quantity * 9.99, 2)
        return {"status": "success", "product_id": product_id, "quantity": quantity, "cart_total": cart["total"]}

    def search_flights(self, destination, date, origin=None, **kw):
        key = f"{origin}|{destination}|{date}"
        table = self.db["flights.json"]["search"]
        if key not in table:
            table[key] = [{
                "flight_id": f"FL{_stable_int(key, 'id', lo=100, hi=999)}",
                "airline": _stable_choice(["United", "Delta", "American", "JetBlue"], key),
                "price": float(_stable_int(key, lo=120, hi=650)), "depart": "09:00", "arrive": "11:45",
            }]
        return {"status": "success", "flights": table[key]}

    def book_flight(self, passenger_name, flight_id, destination=None, date=None, **kw):
        bookings = self.db["flights.json"]["bookings"]
        ref = f"BK{_stable_int(flight_id, passenger_name, lo=1000, hi=9999)}"
        bookings[flight_id] = {"booking_ref": ref, "passenger": passenger_name}
        return {"status": "success", "booking_ref": ref, "flight_id": flight_id, "passenger": passenger_name}

    def calculate_commute(self, origin_address, destination_address, mode="driving", **kw):
        key = f"{origin_address}|{destination_address}|{mode}"
        table = self.db["commute_distances.json"]
        if key not in table:
            table[key] = {"duration_mins": _stable_int(key, lo=8, hi=55),
                           "distance_miles": round(_stable_int(key, "d", lo=10, hi=200) / 10, 1)}
        return {"status": "success", **table[key], "mode": mode}

    def get_card_benefits(self, card_type, **kw):
        table = self.db["cards.json"]
        entry = table.get(card_type.lower())
        if entry is None:
            entry = table[card_type.lower()] = {"benefits": ["1% cashback"]}
        return {"status": "success", "card_type": card_type, "benefits": entry["benefits"]}

    def get_exchange_rate(self, amount, from_currency, to_currency, **kw):
        key = f"{from_currency.upper()}|{to_currency.upper()}"
        table = self.db["exchange_rates.json"]
        rate = table.get(key)
        if rate is None:
            rate = table[key] = round(0.5 + _stable_int(key, lo=0, hi=150) / 100, 4)
        return {"status": "success", "rate": rate, "converted_amount": round(float(amount) * rate, 2)}

    def modify_autopay(self, bill_type, source_account, **kw):
        table = self.db["accounts.json"]
        acct = table.setdefault(source_account, {"balance": 1000.0, "autopay": {}})
        acct["autopay"][bill_type] = True
        return {"status": "success", "autopay_enabled": True, "bill": bill_type, "source": source_account}

    def search_apartments(self, city, bedrooms, max_price, pets_allowed=None, **kw):
        key = f"{city}|{bedrooms}|{max_price}"
        table = self.db["apartments.json"]["search"]
        if key not in table:
            table[key] = [{
                "id": f"APT-{_stable_int(key, lo=1000, hi=9999)}",
                "rent": max(400, int(max_price) - _stable_int(key, "r", lo=20, hi=250)),
                "bedrooms": bedrooms, "pets_allowed": bool(_stable_int(key, "p", lo=0, hi=1)),
                "address": f"{_stable_int(key, 'a', lo=1, hi=999)} Sample St",
            }]
        results = table[key]
        if pets_allowed is not None:
            results = [r for r in results if r["pets_allowed"] == bool(pets_allowed)] or results
        return {"status": "success", "city": city, "results": results}

    def search_products(self, query, max_price=None, **kw):
        table = self.db["products.json"]["search"]
        key = query.lower()
        if key not in table:
            table[key] = [{"product_id": f"PROD-{_stable_int(key, lo=1000, hi=9999)}",
                            "name": f"{query.title()} Standard",
                            "price": _stable_int(key, lo=15, hi=300) + 0.99}]
        results = table[key]
        if max_price is not None:
            results = [r for r in results if r["price"] <= float(max_price)] or results
        return {"status": "success", "products": results}

    def track_order(self, order_id, **kw):
        table = self.db["orders.json"]
        entry = table.get(order_id)
        if entry is None:
            entry = table[order_id] = {
                "status": _stable_choice(["Processing", "Out for delivery", "Delivered", "Shipped"], order_id),
                "eta": "2026-09-01",
            }
        return {"status": "success", "order_id": order_id, "shipping_status": entry["status"], "eta": entry["eta"]}

    def update_identity_doc(self, doc_type, doc_number, **kw):
        self.db["identity_docs.json"][doc_type] = {"masked_number": str(doc_number)[-4:]}
        return {"status": "success", "updated_doc": doc_type, "masked_number": str(doc_number)[-4:]}

    def update_search_filter(self, filter_name, value, **kw):
        self.db["filters.json"][filter_name] = value
        return {"status": "success", "filter_updated": filter_name, "new_value": value}

    # ================= VAB seqdep_cluster (DB-backed) =================

    def location_get_coordinates(self, address, **kw):
        table = self.db["locations.json"]
        entry = table.get(address)
        if entry is None:
            entry = table[address] = {
                "lat": round(30 + _stable_int(address, "lat", lo=0, hi=1500) / 100, 4),
                "lng": round(-125 + _stable_int(address, "lng", lo=0, hi=5500) / 100, 4),
            }
        return {"status": "success", "address": address, **entry}

    def items_search(self, area=None, vendor=None, product=None, **kw):
        table = self.db["items.json"]["search"]
        key = f"{product}|{area}"
        if key not in table:
            table[key] = [{"item_id": f"ITEM-{_stable_int(key, lo=100, hi=999)}",
                            "vendor": (vendor[0] if vendor else "Local Vendor"),
                            "name": str(product), "price": round(5 + _stable_int(key, lo=0, hi=3000) / 100, 2)}]
        return {"status": "success", "items": table[key]}

    def basket_add_item(self, provider_ref, location_ref, product_ref, count=1, **kw):
        basket = self.db["basket.json"]
        basket["items"].append({"provider_ref": provider_ref, "location_ref": location_ref,
                                 "product_ref": product_ref, "count": count})
        basket["total"] = round(basket["total"] + count * 9.99, 2)
        # Report units, not line items: a bare len(items) "basket_size" read as a unit count
        # made models top up with extra calls (count=3 -> size 1 -> count=2 -> size 2 -> count=1).
        return {"status": "success", "added_count": count,
                "total_quantity": sum(i["count"] for i in basket["items"]), "total": basket["total"]}

    def trip_estimate_cost(self, start_location, end_location, **kw):
        key = f"{start_location}|{end_location}"
        table = self.db["trips.json"]
        if key not in table:
            table[key] = {"pricing_id": f"PRICING{_stable_int(key, lo=100, hi=999)}",
                           "estimate": round(6 + _stable_int(key, lo=0, hi=4000) / 100, 2)}
        return {"status": "success", **table[key]}

    # ================= pure/computed (no suggested_db_table) =================

    def vehicle_check_availability(self, start_location, end_location, **kw):
        return {"status": "success", "available": True, "eta_mins": _stable_int(start_location, end_location, lo=2, hi=12)}

    def trip_confirm_booking(self, pricing_id, pickup_location, drop_location, **kw):
        return {"status": "success", "confirmation_id": f"RIDE{_stable_int(pricing_id, lo=1000, hi=9999)}", "pricing_id": pricing_id}

    def get_weather(self, location, **kw):
        table = self.db["weather.json"]
        entry = table.get(location)
        if entry is None:
            entry = table[location] = {"condition": _stable_choice(["Sunny", "Cloudy", "Rainy", "Windy"], location),
                                        "temp_f": _stable_int(location, lo=40, hi=95)}
        return {"status": "success", "location": location, **entry}

    def get_stock_price(self, symbol=None, exchange=None, company_name=None, date=None, **kw):
        key = f"{symbol or company_name}|{exchange}|{date}"
        return {"status": "success", "symbol": symbol or company_name, "exchange": exchange, "date": date,
                "price": round(20 + _stable_int(key, lo=0, hi=48000) / 100, 2)}

    def find_movies(self, title=None, location=None, date=None, **kw):
        return {"status": "success", "results": [{"title": title or "Sample Movie",
                                                    "theater": f"{location or 'Downtown'} Cinema", "showtime": "19:30"}]}

    def book_appointment(self, provider_name, date, time, **kw):
        return {"status": "success", "confirmation_id": f"APPT{_stable_int(provider_name, date, time, lo=1000, hi=9999)}",
                "provider_name": provider_name, "date": date, "time": time}

    def find_provider(self, service_type, location, **kw):
        return {"status": "success", "results": [{"name": f"{location.title()} {service_type.title()} Clinic",
                                                    "rating": round(3.5 + _stable_int(service_type, location, lo=0, hi=15) / 10, 1)}]}

    def find_restaurants(self, cuisine, location, **kw):
        return {"status": "success", "results": [{"name": f"{location.title()} {cuisine.title()} House",
                                                    "rating": round(3.5 + _stable_int(cuisine, location, lo=0, hi=15) / 10, 1)}]}

    def restaurants_2_reserve_restaurant(self, restaurant_name, time, party_size=2, **kw):
        return {"status": "success", "confirmation_id": f"RES{_stable_int(restaurant_name, time, lo=1000, hi=9999)}",
                "restaurant_name": restaurant_name, "time": time, "party_size": party_size}

    def events_3_find_events(self, location, category=None, **kw):
        return {"status": "success", "results": [{"event_id": f"EVT-{_stable_int(location, category, lo=100, hi=999)}",
                                                    "name": f"{location.title()} {category.title() if category else 'Live'} Event",
                                                    "date": "2026-09-05"}]}

    def events_3_buy_event_tickets(self, event_id, quantity=1, **kw):
        return {"status": "success", "confirmation_id": f"TIX{_stable_int(event_id, lo=1000, hi=9999)}",
                "event_id": event_id, "quantity": quantity}

    def hotels_2_search_house(self, location, guests, check_in=None, check_out=None, **kw):
        return {"status": "success", "results": [{"listing_id": f"HOUSE-{_stable_int(location, check_in, lo=100, hi=999)}",
                                                    "location": location, "guests": guests,
                                                    "nightly_rate": _stable_int(location, lo=80, hi=400)}]}

    def hotels_4_search_hotel(self, location, check_in=None, check_out=None, star_rating=None, **kw):
        return {"status": "success", "results": [{"hotel_id": f"HOTEL-{_stable_int(location, check_in, lo=100, hi=999)}",
                                                    "location": location, "star_rating": star_rating,
                                                    "nightly_rate": _stable_int(location, "h", lo=90, hi=450)}]}

    def homes_2_find_home_by_area(self, area, max_price, bedrooms=None, property_type=None, **kw):
        price = max(50000, int(max_price) - _stable_int(area, "p", lo=5000, hi=50000))
        return {"status": "success", "results": [{"home_id": f"HOME-{_stable_int(area, lo=100, hi=999)}", "area": area,
                                                    "price": price, "bedrooms": bedrooms,
                                                    "property_type": property_type or "single-family"}]}

    # ================= implicit-context resolvers (DB-backed, zero-arg) =================

    def get_current_location(self, **kw):
        return {"status": "success", **self.db["session_context.json"]["current_location"]}

    def get_current_date(self, **kw):
        return {"status": "success", "date": self.db["session_context.json"]["current_date"]}

    def get_current_time(self, **kw):
        ctx = self.db["session_context.json"]
        return {"status": "success", "time": ctx["current_time"], "timezone": ctx["timezone"]}

    def get_user_profile(self, **kw):
        return {"status": "success", **self.db["user_profile.json"]}

    TOOLS = {}


# Map tool names (snake_case, identical to the method names above) to their
# implementations; kept explicit so the tool set is visible in one place.
MockToolExecutor.TOOLS = {
    "add_to_cart": MockToolExecutor.add_to_cart,
    "book_flight": MockToolExecutor.book_flight,
    "calculate_commute": MockToolExecutor.calculate_commute,
    "get_card_benefits": MockToolExecutor.get_card_benefits,
    "get_exchange_rate": MockToolExecutor.get_exchange_rate,
    "modify_autopay": MockToolExecutor.modify_autopay,
    "search_apartments": MockToolExecutor.search_apartments,
    "search_flights": MockToolExecutor.search_flights,
    "search_products": MockToolExecutor.search_products,
    "track_order": MockToolExecutor.track_order,
    "update_identity_doc": MockToolExecutor.update_identity_doc,
    "update_search_filter": MockToolExecutor.update_search_filter,
    "location_get_coordinates": MockToolExecutor.location_get_coordinates,
    "items_search": MockToolExecutor.items_search,
    "basket_add_item": MockToolExecutor.basket_add_item,
    "trip_estimate_cost": MockToolExecutor.trip_estimate_cost,
    "vehicle_check_availability": MockToolExecutor.vehicle_check_availability,
    "trip_confirm_booking": MockToolExecutor.trip_confirm_booking,
    "get_weather": MockToolExecutor.get_weather,
    "get_stock_price": MockToolExecutor.get_stock_price,
    "find_movies": MockToolExecutor.find_movies,
    "book_appointment": MockToolExecutor.book_appointment,
    "find_provider": MockToolExecutor.find_provider,
    "find_restaurants": MockToolExecutor.find_restaurants,
    "restaurants_2_reserve_restaurant": MockToolExecutor.restaurants_2_reserve_restaurant,
    "events_3_find_events": MockToolExecutor.events_3_find_events,
    "events_3_buy_event_tickets": MockToolExecutor.events_3_buy_event_tickets,
    "hotels_2_search_house": MockToolExecutor.hotels_2_search_house,
    "hotels_4_search_hotel": MockToolExecutor.hotels_4_search_hotel,
    "homes_2_find_home_by_area": MockToolExecutor.homes_2_find_home_by_area,
    "get_current_location": MockToolExecutor.get_current_location,
    "get_current_date": MockToolExecutor.get_current_date,
    "get_current_time": MockToolExecutor.get_current_time,
    "get_user_profile": MockToolExecutor.get_user_profile,
}

assert set(MockToolExecutor.TOOLS) == {
    t["name"] for t in json.loads((ROOT.parent / "data" / "tool_pool.json").read_text())["tools"]
}, "MockToolExecutor.TOOLS must cover exactly the 34 tools in data/tool_pool.json"
