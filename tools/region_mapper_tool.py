"""
Region Mapper Tool
Maps and queries geographic regions for satellite observations
"""

import asyncio
from geopy.geocoders import Nominatim
from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from geopy.extra.rate_limiter import AsyncRateLimiter
import logging

logger = logging.getLogger(__name__)

async def execute(params):
    """
    Map geographic regions to coordinates for satellite imagery queries.
    
    Args:
        params: Dictionary with:
            - region_name: Name of the geographic region (str, optional)
            - coordinates: [lat, lon] coordinates (list, optional)
            - expand_bbox: Expansion factor for bbox (float, optional)
    
    Returns:
        Dictionary with:
            - status: "success" or "error"
            - region_name: Input region name
            - bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
            - center: Center coordinates as [lat, lon]
            - source: "coordinates" or "geocoded"
            - message: Status message
    """
    
    if not params or not isinstance(params, dict):
        params = {}
    
    region_name = params.get("region_name")
    coordinates = params.get("coordinates")
    expand_bbox = params.get("expand_bbox", 0.0)
    
    if not region_name and not coordinates:
        return {
            "status": "error",
            "error_type": "missing_parameters",
            "message": "Missing required input: Please provide either 'region_name' (e.g., 'Taiwan Strait', 'New York') or 'coordinates' (e.g., [25.0, 121.5])",
            "tool": "region_mapper",
            "params_received": params,
            "required_parameters": {
                "option_1": {"region_name": "string - any location name"},
                "option_2": {"coordinates": "[lat, lon] - latitude and longitude"},
                "optional": {"expand_bbox": "number - expansion factor (e.g., 0.1 for 10%)"}
            },
            "suggested_fix": "Extract the location/region name from the original task and add it to parameters. If the task mentions a specific place (city, region, coordinates), use that as 'region_name' parameter.",
            "examples": [
                {"region_name": "New York"},
                {"region_name": "Taiwan Strait"},
                {"coordinates": [40.7128, -74.0060]},
                {"region_name": "Mediterranean Sea", "expand_bbox": 0.1}
            ]
        }
    
    try:
        if coordinates and isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
            lat, lon = coordinates
            bbox = _create_bbox_from_point(lon, lat, expand_bbox or 0.5)
            return {
                "status": "success",
                "region_name": region_name or f"Point ({lat}, {lon})",
                "bbox": bbox,
                "center": [lat, lon],
                "source": "coordinates",
                "message": "Bounding box created from coordinates",
                "tool": "region_mapper"
            }
        
        if region_name:
            result = await _geocode_region(region_name, expand_bbox)
            return result
        
        return {
            "status": "error",
            "message": "Could not map region with provided parameters",
            "tool": "region_mapper",
            "params_received": params
        }
    
    except Exception as e:
        logger.error(f"Error in region_mapper: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "tool": "region_mapper",
            "params_received": params
        }

async def _geocode_region(region_name, expand_bbox=0.0, max_retries=3):
    """
    Geocode a region name using Nominatim with async support and rate limiting.
    
    Args:
        region_name: Name of region to geocode
        expand_bbox: Optional expansion factor
        max_retries: Maximum retry attempts
        
    Returns:
        Dictionary with geocoding results
    """
    
    for attempt in range(max_retries):
        try:
            async with Nominatim(
                user_agent="eo_satellite_query_tum_research/1.0",
                adapter_factory=AioHTTPAdapter,
                timeout=10
            ) as geolocator:
                
                geocode = AsyncRateLimiter(
                    geolocator.geocode,
                    min_delay_seconds=1.0,
                    max_retries=2
                )
                
                location = await geocode(
                    region_name,
                    exactly_one=True,
                    addressdetails=True
                )
                
                if not location:
                    return {
                        "status": "error",
                        "message": f"Region '{region_name}' not found by geocoder",
                        "region_name": region_name,
                        "tool": "region_mapper"
                    }
                
                if hasattr(location, 'raw') and 'boundingbox' in location.raw:
                    raw_bbox = location.raw['boundingbox']
                    bbox = [
                        float(raw_bbox[2]),
                        float(raw_bbox[0]),
                        float(raw_bbox[3]),
                        float(raw_bbox[1])
                    ]
                    
                    if expand_bbox > 0:
                        bbox = _expand_bbox(bbox, expand_bbox)
                    
                    center = [location.latitude, location.longitude]
                    
                    logger.info(f"Successfully geocoded: {region_name}")
                    return {
                        "status": "success",
                        "region_name": region_name,
                        "bbox": bbox,
                        "center": center,
                        "source": "geocoded",
                        "address": location.address,
                        "message": f"Successfully geocoded region: {region_name}",
                        "tool": "region_mapper"
                    }
                else:
                    bbox = _create_bbox_from_point(
                        location.longitude,
                        location.latitude,
                        expand_bbox or 0.5
                    )
                    center = [location.latitude, location.longitude]
                    
                    logger.warning(f"No bbox in geocoding result for {region_name}, created from point")
                    return {
                        "status": "success",
                        "region_name": region_name,
                        "bbox": bbox,
                        "center": center,
                        "source": "geocoded",
                        "address": location.address,
                        "message": f"Geocoded to point, created bounding box",
                        "tool": "region_mapper"
                    }
        
        except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as e:
            logger.warning(f"Geocoding attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + (asyncio.get_event_loop().time() % 1)
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "status": "error",
                    "message": f"Geocoding failed after {max_retries} attempts: {str(e)}",
                    "region_name": region_name,
                    "tool": "region_mapper"
                }
        
        except Exception as e:
            logger.error(f"Unexpected error during geocoding: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Geocoding error: {str(e)}",
                "region_name": region_name,
                "tool": "region_mapper"
            }
    
    return {
        "status": "error",
        "message": f"Failed to geocode region after {max_retries} attempts",
        "region_name": region_name,
        "tool": "region_mapper"
    }

def _create_bbox_from_point(lon, lat, buffer_degrees=0.5):
    """
    Create a bounding box around a point.
    
    Args:
        lon: Longitude of center point
        lat: Latitude of center point
        buffer_degrees: Buffer distance in degrees
        
    Returns:
        List [min_lon, min_lat, max_lon, max_lat]
    """
    return [
        lon - buffer_degrees,
        lat - buffer_degrees,
        lon + buffer_degrees,
        lat + buffer_degrees
    ]

def _expand_bbox(bbox, expansion_factor):
    """
    Expand a bounding box by a factor.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        expansion_factor: Factor to expand (0.1 = 10% expansion)
        
    Returns:
        Expanded bbox
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    expansion_lon = lon_range * expansion_factor / 2
    expansion_lat = lat_range * expansion_factor / 2
    
    return [
        min_lon - expansion_lon,
        min_lat - expansion_lat,
        max_lon + expansion_lon,
        max_lat + expansion_lat
    ]
