"""Serving units and their gram equivalents, per food category.

Every unit is a *countable* unit (katori, scoop, roti, piece, glass ...) so the
recommendation LLM can only ever emit ``<integer> <unit>`` - never an
uncountable noun like "a bowlful of" or "a little".

Gram values are the mess's own conventions where the printed rules give one:
drinks 200 ml, curd 100 ml, raita 150 ml.
"""
from __future__ import annotations

from collections import OrderedDict

UNIT_TO_GRAMS: dict[str, "OrderedDict[str, float]"] = {
    "rice": OrderedDict([
        ("katori", 90.0),
        ("bowl", 180.0),
        ("plate", 300.0),
        ("ladle", 130.0),
    ]),
    "bread": OrderedDict([
        ("roti", 45.0),
        ("paratha", 80.0),
        ("naan", 90.0),
        ("bhature", 60.0),
        ("bread_slice", 30.0),
    ]),
    "dal": OrderedDict([
        ("katori", 120.0),
        ("bowl", 200.0),
        ("ladle", 100.0),
    ]),
    "curry": OrderedDict([
        ("katori", 120.0),
        ("bowl", 200.0),
        ("piece", 100.0),
        ("ladle", 110.0),
    ]),
    "raita_curd": OrderedDict([
        ("katori", 100.0),
        ("cup", 150.0),
        ("bowl", 200.0),
    ]),
    "chutney_pickle": OrderedDict([
        ("teaspoon", 5.0),
        ("tablespoon", 15.0),
        ("katori", 30.0),
    ]),
    "egg": OrderedDict([
        ("piece", 50.0),
        ("katori", 90.0),
    ]),
    "snack": OrderedDict([
        ("piece", 60.0),
        ("plate", 120.0),
    ]),
    "dessert": OrderedDict([
        ("piece", 60.0),
        ("scoop", 60.0),
        ("bowl", 120.0),
        ("cup", 80.0),
    ]),
    "drink": OrderedDict([
        ("glass", 200.0),
        ("cup", 150.0),
        ("bottle", 500.0),
    ]),
    "salad": OrderedDict([
        ("katori", 80.0),
        ("plate", 150.0),
    ]),
    "fruit": OrderedDict([
        ("piece", 120.0),
        ("bowl", 150.0),
    ]),
    "boilerplate": OrderedDict([
        ("roti", 45.0),
        ("katori", 100.0),
        ("glass", 200.0),
    ]),
}

CATEGORY_DESCRIPTIONS = {
    "rice": "Rice based dishes - plain rice, jeera rice, pulao/pulav, biryani, khichdi, dalia.",
    "bread": "Flatbreads and breads - roti, chapati, butter roti, naan, paratha, bhature, puri.",
    "dal": "Lentils and pulse gravies - dal fry/tadka/makhani, chana dal, arhar dal, chole, rajma.",
    "curry": "Thick vegetable / paneer / chicken gravies served in a katori.",
    "raita_curd": "Curd, dahi, raita and boondi raita (mess serves 100 ml curd, 150 ml raita).",
    "chutney_pickle": "Chutney and pickle accompaniments.",
    "egg": "Whole boiled egg and egg bhurji.",
    "snack": "Fried or steamed breakfast items - dosa, idly, poha, pakoda, sandwich.",
    "dessert": "Sweet items - jalebi rabri, shahi tukda, gulab jamun, ice cream.",
    "drink": "Drinks - lassi, chaas, butter milk, lemon water, juices. Mess serves 200 ml.",
    "salad": "Raw salad, cucumber, carrot, tomato, beetroot, onion, chilli, lemon.",
    "fruit": "Fresh or seasonal fruit.",
    "boilerplate": "Items the mess serves every day regardless of the day of week.",
}

MAX_PORTIONS = 6  # a recommendation never exceeds this many units of one item


def grams(category: str, unit: str, default: float | None = None) -> float | None:
    table = UNIT_TO_GRAMS.get(category, {})
    if unit in table:
        return float(table[unit])
    return default


def allowed_units(category: str) -> list[str]:
    return list(UNIT_TO_GRAMS.get(category, {}).keys())
