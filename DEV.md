## Install Frontend:
1. cp -RPp .env.example .env
2. npm install
    - run `nvm use` if nvm is installed to sync node version

## Install Backend (in uv environment)
1. cd backend
2. uv venv
3. source .venv/bin/activate
4. pip install -r requirements.txt -U

### Debugging missing pip
1. python -m ensurepip --upgrade
2. python -m pip install --upgrade pip


## Startup Frontend:
1. npm run dev

## Startup Backend:
1. cd backend/
    - set virtual environment if needed: source .venv/bin/activate
2. sh dev.sh