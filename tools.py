"""
tools.py — External tool functions for ShaktiAgent.
Maps, government scheme lookup, SIP calculator, helpline directory.
All tools require human approval before execution.
"""
from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger("shakti.tools")

# ── Helpline Directory (always available, no approval needed for lookup) ──

HELPLINES: dict[str, str] = {
    "women_helpline": "181",
    "police": "1091",
    "emergency": "112",
    "child_helpline": "1098",
    "domestic_violence": "181",
    "cyber_crime": "1930",
    "ambulance": "108",
    "janani_express": "102",
}


def get_helplines() -> dict[str, str]:
    """Return the full helpline directory."""
    return HELPLINES


# ── Government Scheme Lookup (mock dict-based) ──

SCHEMES: dict[str, dict[str, str]] = {
    "sukanya": {
        "name": "Sukanya Samriddhi Yojana",
        "type": "Savings",
        "eligibility": "Girl child below 10 years",
        "benefit": "8.2% interest, tax-free under 80C (EEE status)",
        "max_deposit": "₹1,50,000/year",
        "where": "Post Office or authorised bank",
        "url": "https://www.indiapost.gov.in/Financial/pages/content/sukanya-samriddhi-accounts.aspx",
    },
    "jsy": {
        "name": "Janani Suraksha Yojana",
        "type": "Maternal Health",
        "eligibility": "All pregnant women (BPL priority)",
        "benefit": "₹1,400 (rural) / ₹1,000 (urban) cash for institutional delivery",
        "where": "Government health centre",
        "url": "https://nhm.gov.in/index1.php?lang=1&level=3&sublinkid=841&lid=309",
    },
    "pmjay": {
        "name": "Pradhan Mantri Jan Arogya Yojana (Ayushman Bharat)",
        "type": "Health Insurance",
        "eligibility": "Families identified via SECC database",
        "benefit": "Up to ₹5 lakh/year for secondary & tertiary hospitalisation",
        "where": "Empanelled hospitals",
        "url": "https://pmjay.gov.in",
    },
    "pmmvy": {
        "name": "Pradhan Mantri Matru Vandana Yojana",
        "type": "Maternity Benefit",
        "eligibility": "Women aged 19+ for first live birth",
        "benefit": "₹5,000 in 3 instalments",
        "where": "Anganwadi centre / health facility",
        "url": "https://wcd.nic.in/schemes/pradhan-mantri-matru-vandana-yojana",
    },
    "mudra": {
        "name": "Pradhan Mantri Mudra Yojana",
        "type": "Entrepreneurship Loan",
        "eligibility": "Non-corporate, non-farm small/micro enterprises",
        "benefit": "Up to ₹10 lakh, no collateral",
        "where": "Banks, MFIs, NBFCs",
        "url": "https://www.mudra.org.in",
    },
    "standup": {
        "name": "Stand-Up India",
        "type": "Entrepreneurship Loan",
        "eligibility": "SC/ST and women entrepreneurs",
        "benefit": "₹10 lakh to ₹1 crore for greenfield enterprises",
        "where": "Scheduled commercial banks",
        "url": "https://www.standupmitra.in",
    },
    "scss": {
        "name": "Senior Citizens Savings Scheme",
        "type": "Savings",
        "eligibility": "Age 60+ (55+ for VRS)",
        "benefit": "8.2% interest paid quarterly, 80C benefit",
        "max_deposit": "₹30,00,000",
        "where": "Post Office or authorised bank",
        "url": "https://www.indiapost.gov.in/Financial/pages/content/scss.aspx",
    },
    "nps": {
        "name": "National Pension System",
        "type": "Pension",
        "eligibility": "Indian citizens aged 18-70",
        "benefit": "Tax benefits under 80CCD (up to ₹2 lakh), market-linked returns",
        "where": "NPS POPs (Point of Presence)",
        "url": "https://www.npscra.nsdl.co.in",
    },
    "apy": {
        "name": "Atal Pension Yojana",
        "type": "Pension",
        "eligibility": "Age 18-40, unorganised sector",
        "benefit": "Guaranteed ₹1,000-5,000/month pension after 60",
        "where": "Banks",
        "url": "https://www.npscra.nsdl.co.in/scheme-details.php",
    },
    "pragati": {
        "name": "AICTE Pragati Scholarship",
        "type": "Scholarship",
        "eligibility": "Girls in technical education, family income <₹8 lakh",
        "benefit": "₹50,000/year tuition + ₹2,000/month",
        "where": "scholarships.gov.in",
        "url": "https://scholarships.gov.in",
    },
    "pmkvy": {
        "name": "Pradhan Mantri Kaushal Vikas Yojana",
        "type": "Skill Training",
        "eligibility": "All citizens, no age bar for most courses",
        "benefit": "Free skill training + certification + placement support",
        "where": "PMKVY training centres",
        "url": "https://www.pmkvyofficial.org",
    },
}


def lookup_scheme(keyword: str) -> dict[str, Any]:
    """Look up a government scheme by keyword. Returns scheme details or not-found."""
    keyword_lower = keyword.lower().strip()
    for key, scheme in SCHEMES.items():
        if keyword_lower in key or keyword_lower in scheme["name"].lower():
            return {"found": True, **scheme}
    return {"found": False, "message": f"No scheme found for '{keyword}'. Try: {', '.join(SCHEMES.keys())}"}


# ── SIP Calculator ──

def calculate_sip(
    monthly_amount: float,
    annual_return_pct: float,
    years: int,
) -> dict[str, Any]:
    """
    Calculate SIP maturity value using compound interest formula.
    M = P × [{(1+r)^n - 1} / r] × (1+r)
    where P = monthly amount, r = monthly rate, n = total months.
    """
    if monthly_amount <= 0 or annual_return_pct <= 0 or years <= 0:
        return {"error": "All inputs must be positive numbers."}

    monthly_rate = annual_return_pct / 100 / 12
    total_months = years * 12
    total_invested = monthly_amount * total_months

    maturity = monthly_amount * (((1 + monthly_rate) ** total_months - 1) / monthly_rate) * (1 + monthly_rate)
    wealth_gained = maturity - total_invested

    return {
        "monthly_sip": f"₹{monthly_amount:,.0f}",
        "annual_return": f"{annual_return_pct}%",
        "duration_years": years,
        "total_invested": f"₹{total_invested:,.0f}",
        "estimated_maturity": f"₹{maturity:,.0f}",
        "wealth_gained": f"₹{wealth_gained:,.0f}",
    }


# ── Google Maps — Find Nearby (mock if no API key) ──

MAPS_API_KEY = os.getenv("MAPS_API_KEY", "")

MOCK_CLINICS = [
    {"name": "City Women's Hospital", "address": "MG Road, Bengaluru", "rating": 4.3, "phone": "+91-80-12345678"},
    {"name": "Shakti Maternity Clinic", "address": "Koramangala, Bengaluru", "rating": 4.5, "phone": "+91-80-87654321"},
    {"name": "Government Primary Health Centre", "address": "Jayanagar, Bengaluru", "rating": 3.9, "phone": "+91-80-11223344"},
]

MOCK_POLICE = [
    {"name": "Koramangala Police Station", "address": "80 Feet Road, Koramangala", "phone": "1091"},
    {"name": "Women's Help Desk — MG Road", "address": "MG Road Metro Station", "phone": "181"},
]


async def find_nearby_clinic(latitude: float = 12.9716, longitude: float = 77.5946) -> list[dict[str, Any]]:
    """Find nearby clinics/hospitals using Google Maps Places API. Falls back to mock data."""
    if not MAPS_API_KEY:
        logger.warning("MAPS_API_KEY not set — returning mock clinic data")
        return MOCK_CLINICS

    try:
        import googlemaps  # type: ignore
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        results = gmaps.places_nearby(
            location=(latitude, longitude),
            radius=5000,
            type="hospital",
            keyword="women clinic maternity",
        )
        places = []
        for place in results.get("results", [])[:5]:
            places.append({
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "rating": place.get("rating", "N/A"),
                "open_now": place.get("opening_hours", {}).get("open_now", "Unknown"),
            })
        return places if places else MOCK_CLINICS
    except Exception as e:
        logger.error(f"Maps API error: {e}")
        return MOCK_CLINICS


async def find_nearby_police(latitude: float = 12.9716, longitude: float = 77.5946) -> list[dict[str, Any]]:
    """Find nearby police stations. Falls back to mock data."""
    if not MAPS_API_KEY:
        logger.warning("MAPS_API_KEY not set — returning mock police data")
        return MOCK_POLICE

    try:
        import googlemaps  # type: ignore
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        results = gmaps.places_nearby(
            location=(latitude, longitude),
            radius=5000,
            type="police",
        )
        places = []
        for place in results.get("results", [])[:5]:
            places.append({
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "rating": place.get("rating", "N/A"),
            })
        return places if places else MOCK_POLICE
    except Exception as e:
        logger.error(f"Maps API error: {e}")
        return MOCK_POLICE


# ── Tool Registry (for human oversight gate) ──

TOOL_REGISTRY: dict[str, Any] = {
    "find_nearby_clinic": find_nearby_clinic,
    "find_nearby_police": find_nearby_police,
    "lookup_scheme": lookup_scheme,
    "calculate_sip": calculate_sip,
    "get_helplines": get_helplines,
}

# ── Advanced tools (loaded lazily to avoid import-time failures) ──
_advanced_tools_loaded = False

def _load_advanced_tools():
    """Lazily load advanced tools from advanced_tools.py."""
    global _advanced_tools_loaded
    if _advanced_tools_loaded:
        return
    try:
        from advanced_tools import ADVANCED_TOOL_REGISTRY
        TOOL_REGISTRY.update(ADVANCED_TOOL_REGISTRY)
        _advanced_tools_loaded = True
        logger.info(f"Loaded {len(ADVANCED_TOOL_REGISTRY)} advanced tools")
    except Exception as e:
        logger.warning(f"Advanced tools not loaded: {e}")


async def execute_tool(tool_name: str, params: dict[str, Any]) -> Any:
    """Execute a registered tool by name. Used after human approval."""
    # Load advanced tools on first call
    _load_advanced_tools()
    
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}

    func = TOOL_REGISTRY[tool_name]
    try:
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return await func(**params)
        return func(**params)
    except Exception as e:
        logger.error(f"Tool execution error [{tool_name}]: {e}")
        return {"error": str(e)}
