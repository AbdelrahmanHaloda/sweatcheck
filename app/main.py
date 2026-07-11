"""
SweatCheck - Weather AI API
A weather API that provides human-friendly advice based on weather data.
"""

import os
import time
import logging
from typing import Optional, List
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()       # ← This reads the .env file

# HTTP client for calling OpenWeatherMap
import httpx

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ---------- Pydantic Models (Data Validation) ----------

class RootResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: Dict[str, str]

class CurrentWeatherResponse(BaseModel):
    """Response model for current weather"""
    city: str
    country: str
    temperature: float
    feels_like: float
    humidity: int
    pressure: int
    wind_speed: float
    condition: str
    description: str
    timestamp_readable: str
    advice: str
    heat_index: float

class AdviceResponse(BaseModel):
    """Response model for advice endpoint"""
    city: str
    current_advice: str
    best_time: str
    worst_time: str
    recommendation: str

# ---------- Helper Functions ----------

async def fetch_weather_data(city: str, units: str = "metric", endpoint: str = "weather"):
    """
    Fetch weather data from OpenWeatherMap.
    """
    if not OPENWEATHER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key not configured. Set OPENWEATHER_API_KEY environment variable."
        )
    
    url = f"{OPENWEATHER_BASE_URL}/{endpoint}"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": units
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        elif response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid API key")
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Weather API error: {response.status_code}"
            )
        
        return response.json()
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Weather API timeout")
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

def compute_heat_index(temp_celsius: float, humidity: int) -> float:
    """
    Compute heat index (what the temperature actually feels like).
    Formula from NOAA (National Oceanic and Atmospheric Administration).
    
    Heat Index is only valid for temperatures above 27°C (80°F).
    Below that, it's just the temperature.
    """
    if temp_celsius < 27:
        return temp_celsius
    
    # Convert to Fahrenheit for the NOAA formula
    temp_f = (temp_celsius * 9/5) + 32
    humidity_percent = humidity / 100.0
    
    hi = (
        -42.379 +
        2.04901523 * temp_f +
        10.14333127 * humidity_percent -
        0.22475541 * temp_f * humidity_percent -
        0.00683783 * temp_f * temp_f -
        0.05481717 * humidity_percent * humidity_percent +
        0.00122874 * temp_f * temp_f * humidity_percent +
        0.00085282 * temp_f * humidity_percent * humidity_percent -
        0.00000199 * temp_f * temp_f * humidity_percent * humidity_percent
    )
    
    # Convert back to Celsius
    return (hi - 32) * 5/9

def generate_weather_advice(temp_celsius: float, feels_like_celsius: float, humidity: int, condition: str) -> str:
    """
    Generate human-friendly advice based on weather conditions.
    This is our "AI" for now - rule-based advice engine.
    """
    advice_parts = []
    
    # Temperature advice
    if temp_celsius >= 40:
        advice_parts.append("☀️ EXTREME HEAT! Stay indoors if possible")
    elif temp_celsius >= 35:
        advice_parts.append("🔥 Very hot - limit outdoor activities")
    elif temp_celsius >= 30:
        advice_parts.append("🌡️ Hot - stay hydrated and seek shade")
    elif temp_celsius >= 25:
        advice_parts.append("☀️ Warm and pleasant")
    elif temp_celsius >= 20:
        advice_parts.append("🌤️ Nice weather - enjoy!")
    elif temp_celsius >= 15:
        advice_parts.append("🌥️ Comfortable - good for outdoor activities")
    elif temp_celsius >= 10:
        advice_parts.append("🍂 A bit cool - bring a light jacket")
    elif temp_celsius >= 5:
        advice_parts.append("🧥 Cool - wear a warm jacket")
    else:
        advice_parts.append("🥶 COLD! Layer up")
    
    # Humidity advice
    if humidity >= 80:
        advice_parts.append("💧 Very humid - sweat won't evaporate, you'll feel sticky")
    elif humidity >= 60:
        advice_parts.append("💧 Humid - you might feel warmer than it actually is")
    elif humidity <= 30:
        advice_parts.append("🌵 Dry air - stay hydrated")
    
    # Feels like vs actual temperature
    if feels_like_celsius > temp_celsius + 3:
        advice_parts.append(f"⚠️ Feels like {feels_like_celsius:.0f}°C due to humidity")
    
    # Condition advice
    if "rain" in condition.lower():
        advice_parts.append("☔ Rain expected - bring an umbrella")
    elif "thunder" in condition.lower():
        advice_parts.append("⛈️ Thunderstorms possible - stay safe indoors")
    elif "snow" in condition.lower():
        advice_parts.append("❄️ Snow expected - be careful on roads")
    elif "clear" in condition.lower() or "sunny" in condition.lower():
        if temp_celsius > 30:
            advice_parts.append("☀️ Sun protection recommended")
    
    return " | ".join(advice_parts)

# ---------- FastAPI App ----------

app = FastAPI(
    title="SweatCheck Weather AI",
    description="Weather API that provides human-friendly advice",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Endpoints ----------

@app.get("/", response_model=RootResponse, tags=["System"])
async def root():
    """Root endpoint - API information"""
    return RootResponse(
        name="SweatCheck Weather AI",
        version="1.0.0",
        description="Get human-friendly weather advice",
        endpoints={
            "/weather": "Get current weather with advice",
            "/weather/forecast": "Get forecast with best times",
            "/weather/advice": "Get advice only",
            "/health": "Health check"
        }
    )

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    now = datetime.now()
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "timestamp_readable": now.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/weather", response_model=CurrentWeatherResponse, tags=["Weather"])
async def get_current_weather(
    city: str = Query(..., description="City name (e.g., Cairo, London)"),
    units: str = Query("metric", description="metric=Celsius, imperial=Fahrenheit")
):
    """Get current weather with human-friendly advice."""
    
    # Fetch data
    data = await fetch_weather_data(city, units, "weather")
    
    # Extract data
    temp_celsius = data['main']['temp']
    feels_like_celsius = data['main']['feels_like']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    wind_speed = data['wind']['speed']
    condition = data['weather'][0]['main']
    description = data['weather'][0]['description']
    
    # Compute heat index
    heat_index_celsius = compute_heat_index(temp_celsius, humidity)
    
    # Generate advice
    advice = generate_weather_advice(temp_celsius, feels_like_celsius, humidity, condition)
    
    # Convert to requested units
    if units == "imperial":
        temp_display = (temp_celsius * 9/5) + 32
        feels_like_display = (feels_like_celsius * 9/5) + 32
        heat_index_display = (heat_index_celsius * 9/5) + 32
    else:
        temp_display = temp_celsius
        feels_like_display = feels_like_celsius
        heat_index_display = heat_index_celsius
    
    # Timestamps
    now = datetime.now()
    timestamp_readable = now.strftime("%Y-%m-%d %H:%M:%S")
    
    return CurrentWeatherResponse(
        city=data['name'],
        country=data['sys']['country'],
        temperature=temp_display,
        feels_like=feels_like_display,
        humidity=humidity,
        pressure=pressure,
        wind_speed=wind_speed,
        condition=condition,
        description=description,
        timestamp_readable=timestamp_readable,
        advice=advice,
        heat_index=heat_index_display
    )

@app.get("/weather/advice", response_model=AdviceResponse, tags=["Weather"])
async def get_weather_advice(
    city: str = Query(..., description="City name"),
    units: str = Query("metric", description="metric=Celsius, imperial=Fahrenheit")
):
    """Get just the advice without all the data."""
    
    data = await fetch_weather_data(city, units, "weather")
    
    temp_celsius = data['main']['temp']
    feels_like_celsius = data['main']['feels_like']
    humidity = data['main']['humidity']
    condition = data['weather'][0]['main']
    
    advice = generate_weather_advice(temp_celsius, feels_like_celsius, humidity, condition)
    
    # Simple recommendation
    if temp_celsius > 35:
        recommendation = "☀️ Avoid going out between 11 AM and 4 PM. Stay hydrated!"
    elif temp_celsius > 30:
        recommendation = "🌤️ Go out in the morning or evening. Avoid midday sun."
    elif temp_celsius > 25:
        recommendation = "☀️ Enjoy the weather! It's warm but comfortable."
    elif temp_celsius > 20:
        recommendation = "🌤️ Great day to be outside!"
    elif temp_celsius > 15:
        recommendation = "🍂 Good for outdoor activities. Bring a light jacket."
    else:
        recommendation = "🧥 Bundle up! It's cold out there."
    
    return AdviceResponse(
        city=data['name'],
        current_advice=advice,
        best_time="Check /weather/forecast for details",
        worst_time="Check /weather/forecast for details",
        recommendation=recommendation
    )

# ---------- If Running Directly ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)