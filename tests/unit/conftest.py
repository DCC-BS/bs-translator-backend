import os

# Container.app_config evaluates AppConfig.from_env() when a router module is imported,
# so the required variables need a value before any test module imports one.
for _name, _value in {
    "LLM_URL": "http://localhost:8001/v1",
    "LLM_API_KEY": "none",
    "LLM_MODEL": "test/test-model",
    "CLIENT_URL": "http://localhost:3000",
    "DOCLING_URL": "http://localhost:8004/v1",
    "DOCLING_API_KEY": "none",
    "HMAC_SECRET": "test-secret",
    "WHISPER_URL": "http://localhost:50001/v1",
}.items():
    os.environ.setdefault(_name, _value)
