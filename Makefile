.PHONY: dev-back dev-front dev

dev-back:
	uvicorn backend.app.main:app --reload --port 8000

dev-front:
	npm run dev --prefix frontend

# Fire both (logs interleaved). Press Ctrl+C to stop.
dev:
	(uvicorn backend.app.main:app --reload --port 8000 &) \
	&& (npm run dev --prefix frontend)
