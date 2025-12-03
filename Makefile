.PHONY: dev-back dev-front dev-stream dev

dev-back:
	uvicorn backend.app.main:app --reload --port 8000

dev-front:
	npm run dev --prefix frontend

dev-stream:
	cd streamlit_app && streamlit run app.py

# Fire all three (logs interleaved). Press Ctrl+C to stop.
dev:
	(uvicorn backend.app.main:app --reload --port 8000 &) \
	&& (npm run dev --prefix frontend &) \
	&& (cd streamlit_app && streamlit run app.py)
