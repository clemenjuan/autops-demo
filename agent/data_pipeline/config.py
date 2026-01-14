import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/autops_db')
KEEPTRACK_API_URL = os.getenv('KEEPTRACK_API_URL', 'https://api.keeptrack.space/v2/sats')
API_PORT = int(os.getenv('API_PORT', 8000))
API_HOST = os.getenv('API_HOST', '0.0.0.0')
SYNC_INTERVAL_HOURS = int(os.getenv('SYNC_INTERVAL_HOURS', 1))
SYNC_TIMEOUT_SECONDS = int(os.getenv('SYNC_TIMEOUT_SECONDS', 30))
