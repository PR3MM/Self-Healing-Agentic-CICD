# Self Healing CI/CD Pipeline with Todo API

A demonstration of an AI-powered self-healing CI/CD pipeline that automatically detects test failures and applies fixes using Gemini AI.

## Project Overview

This project showcases a complete backend Todo API integrated with a self-healing CI/CD pipeline. The pipeline:
- Detects test failures automatically
- Uses Gemini AI to analyze the root cause
- Generates and applies fixes
- Creates pull requests for human review
- Supports human-in-the-loop approval

## Todo API - Backend Service

A FastAPI-based Todo application with complete CRUD operations.

### Features
- ✅ Create, Read, Update, Delete todos
- ✅ In-memory storage for fast local testing
- ✅ Input validation using Pydantic
- ✅ RESTful API design
- ✅ Comprehensive test coverage
- ✅ Health check endpoint

### API Endpoints

#### Get All Todos
```
GET /todos
Response: [
  {
    "id": 1,
    "title": "Learn FastAPI",
    "description": "Complete the tutorial",
    "completed": false,
    "created_at": "2026-06-06T10:30:00"
  }
]
```

#### Create Todo
```
POST /todos
Body: {
  "title": "New Todo",
  "description": "Optional description",
  "completed": false
}
Response: { ...created todo with id... }
```

#### Get Specific Todo
```
GET /todos/{todo_id}
Response: { ...todo object... }
```

#### Update Todo
```
PUT /todos/{todo_id}
Body: {
  "title": "Updated title",
  "completed": true
}
Response: { ...updated todo... }
```

#### Delete Todo
```
DELETE /todos/{todo_id}
Response: { "message": "Todo deleted successfully" }
```

#### Delete All Todos
```
DELETE /todos
Response: { "message": "All todos deleted successfully" }
```

#### Health Check
```
GET /health
Response: { "status": "healthy" }
```

## Running the Application

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API Server
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Run Tests
```bash
pytest tests/test_todos.py -v
```

## Self-Healing Pipeline Architecture

The agentic.py script implements a LangGraph-based state machine that:
1. Monitors CI/CD pipeline failures
2. Analyzes test output and code
3. Uses Gemini AI to generate fixes
4. Validates fixes with sandbox testing
5. Creates pull requests with approved fixes

### Components
- **agentic.py**: Main self-healing agent orchestrator
- **main.py**: Todo API application
- **sandbox.py**: Isolated test execution environment using Docker
- **tests/test_todos.py**: Comprehensive test suite

## Technologies Used
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Lightweight database
- **LangGraph**: Agent orchestration framework
- **Gemini AI**: Code analysis and fix generation
- **PyGithub**: GitHub integration
- **Docker**: Sandbox environment for safe testing

## Demo Scenarios

The self-healing pipeline excels at:
- Fixing simple logic errors (wrong operators, incorrect conditions)
- Updating API endpoint implementations
- Resolving dependency conflicts
- Auto-generating fixes that pass all tests
- Creating human-reviewable pull requests