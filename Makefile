COMPOSE_FILES=$(shell bash check_gpu.sh)

.PHONY: build up down logs clean setup

# Create .env file from example if it doesn't exist
setup:
	@chmod +x check_gpu.sh
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template. Please edit it with your HuggingFace token."; \
	else \
		echo ".env file already exists. Skipping creation."; \
	fi

# Build with HF token
build: setup
	@echo "Building Docker image with HuggingFace token from .env file..."
	docker compose $(COMPOSE_FILES) build

up: setup
	docker compose $(COMPOSE_FILES) up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
