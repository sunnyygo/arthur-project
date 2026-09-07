import os

# Konfigurasi model untuk UQ Agent.
# Key gateway diambil dari env .env Hermes coder (tidak di-hardcode di sini).

BASE_URL = os.environ.get("UQ_LLM_BASE_URL", "https://je.jerouter.web.id/v1")
API_KEY = os.environ.get("UQ_LLM_API_KEY", "")
MODEL = os.environ.get("UQ_LLM_MODEL", "glm-5.3-flash")
