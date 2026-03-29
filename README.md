---
title: Meta-Pytorch-Openenv
emoji: 🦀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# SQL / Data Cleaning Sandbox 

An **OpenEnv**-compliant environment where AI agents clean messy SQLite databases
using SQL queries and Python code.

## Overview

| Feature | Details |
|---|---|
| **Interface** | `step()` / `reset()` / `state()` |
| **Action space** | `{ tool: "sql" \| "python", command: "..." }` |
| **Observation** | `{ output, error, current_step, max_steps, task_description }` |
| **Reward** | 0.0 - 1.0 with **partial progress signals** |
| **Tasks** | 3 (easy, medium, hard) |

## Tasks

### Easy - Data Triage
> Find the total revenue from the `sales` table for January 2024.

**Grader**: Checks if the computed total matches the expected float value (1000.00).

### Medium - Data Cleaning
> Fix duplicate emails, NULL ages, and uppercase emails in the `users` table.

**Grader**: Partial scoring:
- 0.3 for all emails lowercase
- 0.4 for no duplicate emails
- 0.3 for no NULL ages

### Hard - Schema Migration
> Normalize `flat_orders` into `customers` + `orders` tables with foreign keys.

**Grader**: Partial scoring:
- 0.2 for correct `customers` schema
- 0.2 for correct `orders` schema
- 0.2 for 4 unique customers
- 0.2 for 6 orders migrated
- 0.2 for valid FK integrity

## Quick Start

### Local Development

1. **Clone and Install**
```bash
# Clone the repository
git clone https://github.com/shreyas231219/Meta-Pytorch-Openenv.git
cd Meta-Pytorch-Openenv

# Install dependencies
pip install -e .
```

2. **Run the Server**
The server will default to port **7860**. 

**Bash (Linux/macOS):**
```bash
TASK_ID=easy python -m server.app
```

**PowerShell (Windows):**
```powershell
$env:TASK_ID='easy'
python -m server.app
```

### Docker (Hugging Face Spaces Ready)

```bash
# Build
docker build -t sql-sandbox:latest .

# Run on HF Spaces default port 7860
docker run -p 7860:7860 sql-sandbox:latest
```

## Baseline Inference

Runs GPT-4o on all three tasks and prints reproducible scores. 

```powershell
# For local testing in PowerShell (Windows)
$env:HF_TOKEN='sk-...'
$env:MODEL_NAME='gpt-4o'
python inference.py --url http://localhost:7860
```

## Project Structure

```
.
├── Dockerfile              # Root Dockerfile for HF Spaces
├── openenv.yaml            # OpenEnv manifest (port 7860)
├── pyproject.toml           # Package dependencies
├── inference.py            # baseline inference script
├── inference_groq.py       # groq inference script
├── README.md               # This file
├── client.py               # EnvClient helper
├── models.py               # Action & Observation models
└── server/
    ├── app.py              # FastAPI server entry point
    └── environment.py      # Core environment logic + graders
```
