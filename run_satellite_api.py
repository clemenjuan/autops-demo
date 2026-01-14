import uvicorn
from agent.api.main import app, init_db
from agent.data_pipeline.config import DATABASE_URL, API_HOST, API_PORT

if __name__ == '__main__':
    init_db(DATABASE_URL)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
