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

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()       # ← This reads the .env file

# HTTP client for calling OpenWeatherMap
import httpx

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
# Get API key from environment variable (never hardcode!)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ---------- Pydantic Models (Data Validation) ----------

class WeatherRequest(BaseModel):
    """Request model for weather endpoint"""
    city: str
    units: str = "metric"  # metric = Celsius, imperial = Fahrenheit

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
    timestamp: int
    advice: str
    heat_index: Optional[float] = None

class ForecastResponse(BaseModel):
    """Response model for forecast"""
    city: str
    country: str
    hourly: List[dict]
    daily_advice: str

class AdviceResponse(BaseModel):
    """Response model for advice endpoint"""
    city: str
    current_advice: str
    best_time: str
    worst_time: str
    recommendation: str

# ---------- FastAPI App ----------

app = FastAPI(
    title="SweatCheck Weather AI",
    description="Weather API that provides human-friendly advice",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS (so browsers can access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helper Functions ----------

def compute_heat_index(temp_celsius: float, humidity: int) -> float:
    """
    Compute heat index (what the temperature actually feels like).
    Formula from NOAA (National Oceanic and Atmospheric Administration).
    
    Heat Index is only valid for temperatures above 27°C (80°F).
    Below that, it's just the temperature.
    """
    if temp_celsius < 27:
        return temp_celsius
    
    # Convert to Fahrenheit for the formula
    temp_f = (temp_celsius * 9/5) + 32
    humidity_percent = humidity / 100.0
    
    # NOAA Heat Index formula
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

def generate_weather_advice(temp: float, feels_like: float, humidity: int, condition: str) -> str:
    """
    Generate human-friendly advice based on weather conditions.
    This is our "AI" for now - rule-based advice engine.
    """
    advice_parts = []
    
    # Temperature advice
    if temp >= 40:
        advice_parts.append("☀️ EXTREME HEAT! Stay indoors if possible")
    elif temp >= 35:
        advice_parts.append("🔥 Very hot - limit outdoor activities")
    elif temp >= 30:
        advice_parts.append("🌡️ Hot - stay hydrated and seek shade")
    elif temp >= 25:
        advice_parts.append("☀️ Warm and pleasant")
    elif temp >= 20:
        advice_parts.append("🌤️ Nice weather - enjoy!")
    elif temp >= 15:
        advice_parts.append("🌥️ Comfortable - good for outdoor activities")
    elif temp >= 10:
        advice_parts.append("🍂 A bit cool - bring a light jacket")
    elif temp >= 5:
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
    if feels_like > temp + 3:
        advice_parts.append(f"⚠️ Feels like {feels_like:.0f}°C due to humidity")
    
    # Condition advice
    if "rain" in condition.lower() or "drizzle" in condition.lower():
        advice_parts.append("☔ Rain expected - bring an umbrella")
    elif "thunder" in condition.lower():
        advice_parts.append("⛈️ Thunderstorms possible - stay safe indoors")
    elif "snow" in condition.lower():
        advice_parts.append("❄️ Snow expected - be careful on roads")
    elif "clear" in condition.lower() or "sunny" in condition.lower():
        if temp > 30:
            advice_parts.append("☀️ Sun protection recommended")
    
    return " | ".join(advice_parts)

def find_best_time(hourly_data: List[dict]) -> dict:
    """
    Find the best and worst times for outdoor activities.
    Looks for lowest heat index / best conditions.
    """
    if not hourly_data:
        return {"best": "No data", "worst": "No data"}
    
    best_temp = float('inf')
    worst_temp = float('-inf')
    best_time = ""
    worst_time = ""
    
    for hour in hourly_data[:24]:  # Check next 24 hours
        time_str = datetime.fromtimestamp(hour['dt']).strftime('%H:%M')
        temp = hour['main']['temp']
        humidity = hour['main']['humidity']
        feels_like = compute_heat_index(temp, humidity)
        
        # Find best time (coolest feels-like)
        if feels_like < best_temp:
            best_temp = feels_like
            best_time = time_str
        
        # Find worst time (hottest feels-like)
        if feels_like > worst_temp:
            worst_temp = feels_like
            worst_time = time_str
    
    return {
        "best": f"{best_time} (feels like {best_temp:.0f}°C)",
        "worst": f"{worst_time} (feels like {worst_temp:.0f}°C)"
    }

# ---------- API Endpoints ----------

@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "SweatCheck Weather AI",
        "version": "1.0.0",
        "description": "Get human-friendly weather advice",
        "endpoints": {
            "/weather": "Get current weather with advice",
            "/weather/forecast": "Get forecast with best times",
            "/weather/advice": "Get advice only",
            "/health": "Health check"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": int(time.time())
    }

@app.get("/weather", response_model=CurrentWeatherResponse, tags=["Weather"])
async def get_current_weather(
    city: str = Query(..., description="City name (e.g., Cairo, London)"),
    units: str = Query("metric", description="metric=Celsius, imperial=Fahrenheit")
):
    """
    Get current weather with human-friendly advice.
    
    Example: /weather?city=Cairo&units=metric
    """
    # Validate API key
    if not OPENWEATHER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key not configured. Set OPENWEATHER_API_KEY environment variable."
        )
    
    # Build the URL for OpenWeatherMap
    url = f"{OPENWEATHER_BASE_URL}/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": units
    }
    
    try:
        # Make the HTTP request to OpenWeatherMap
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
        
        # Handle errors
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        elif response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid API key")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Weather API error")
        
        data = response.json()
        
        # Extract weather data
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        feels_like = data['main']['feels_like']
        pressure = data['main']['pressure']
        wind_speed = data['wind']['speed']
        condition = data['weather'][0]['main']
        description = data['weather'][0]['description']
        
        # Compute heat index (what it actually feels like)
        heat_index = compute_heat_index(temp, humidity)
        
        # Generate advice
        advice = generate_weather_advice(temp, feels_like, humidity, condition)
        
        return CurrentWeatherResponse(
            city=data['name'],
            country=data['sys']['country'],
            temperature=temp,
            feels_like=feels_like,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            condition=condition,
            description=description,
            timestamp=int(time.time()),
            advice=advice,
            heat_index=heat_index
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Weather API timeout")
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/weather/forecast", tags=["Weather"])
async def get_weather_forecast(
    city: str = Query(..., description="City name"),
    units: str = Query("metric", description="metric=Celsius, imperial=Fahrenheit")
):
    """
    Get weather forecast with best time recommendations.
    
    Example: /weather/forecast?city=Cairo&units=metric
    """
    if not OPENWEATHER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key not configured"
        )
    
    url = f"{OPENWEATHER_BASE_URL}/forecast"
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
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Weather API error")
        
        data = response.json()
        
        # Find best and worst times
        best_times = find_best_time(data['list'])
        
        # Generate daily advice
        hourly = []
        for item in data['list'][:8]:  # 8 x 3-hour intervals = 24 hours
            temp = item['main']['temp']
            humidity = item['main']['humidity']
            feels_like = compute_heat_index(temp, humidity)
            
            hourly.append({
                "time": datetime.fromtimestamp(item['dt']).strftime('%H:%M'),
                "temperature": temp,
                "feels_like": feels_like,
                "humidity": humidity,
                "condition": item['weather'][0]['main']
            })
        
        return {
            "city": data['city']['name'],
            "country": data['city']['country'],
            "hourly": hourly,
            "daily_advice": f"Best time: {best_times['best']} | Worst time: {best_times['worst']}"
        }
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Weather API timeout")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/weather/advice", response_model=AdviceResponse, tags=["Weather"])
async def get_weather_advice(
    city: str = Query(..., description="City name"),
    units: str = Query("metric", description="metric=Celsius, imperial=Fahrenheit")
):
    """
    Get just the advice without all the data.
    
    Example: /weather/advice?city=Cairo&units=metric
    """
    if not OPENWEATHER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key not configured"
        )
    
    url = f"{OPENWEATHER_BASE_URL}/weather"
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
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Weather API error")
        
        data = response.json()
        
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        feels_like = data['main']['feels_like']
        condition = data['weather'][0]['main']
        
        # Generate advice
        current_advice = generate_weather_advice(temp, feels_like, humidity, condition)
        
        # Get forecast for best times
        # (We're keeping it simple here - in production, we'd call forecast too)
        best_time = "Check forecast endpoint for details"
        worst_time = "Check forecast endpoint for details"
        
        # Simple recommendation
        if temp > 35:
            recommendation = "☀️ Avoid going out between 11 AM and 4 PM. Stay hydrated!"
        elif temp > 30:
            recommendation = "🌤️ Go out in the morning or evening. Avoid midday sun."
        elif temp > 25:
            recommendation = "☀️ Enjoy the weather! It's warm but comfortable."
        elif temp > 20:
            recommendation = "🌤️ Great day to be outside!"
        elif temp > 15:
            recommendation = "🍂 Good for outdoor activities. Bring a light jacket."
        else:
            recommendation = "🧥 Bundle up! It's cold out there."
        
        return AdviceResponse(
            city=data['name'],
            current_advice=current_advice,
            best_time=best_time,
            worst_time=worst_time,
            recommendation=recommendation
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Weather API timeout")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ---------- If Running Directly ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)