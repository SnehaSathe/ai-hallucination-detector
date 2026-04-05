FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies (cached layer)
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple
# Now copy rest of code
COPY . .

EXPOSE 8080
EXPOSE 8501



CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
