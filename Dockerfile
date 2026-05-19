# HelpfulBatBot Dockerfile
# Multi-stage build for smaller final image

FROM python:3.11-slim as builder

# Install git (needed for cloning content repositories)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install git in final image (needed at runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY HelpfulBat_app.py .
COPY content_manager.py .
COPY content_sources.yaml .
COPY interaction_logger.py .
COPY ask.py .
COPY start_bot.sh .
COPY demo.sh .

# Create content cache directory
RUN mkdir -p /app/content_cache

# Make scripts executable
RUN chmod +x start_bot.sh demo.sh ask.py

# Add local Python packages to PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start the bot
CMD ["python", "HelpfulBat_app.py"]
