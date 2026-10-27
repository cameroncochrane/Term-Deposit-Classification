# AWS Lambda entrypoint. Wraps the FastAPI app for API Gateway proxy integration.

from mangum import Mangum

from api.main import app

handler = Mangum(app)
