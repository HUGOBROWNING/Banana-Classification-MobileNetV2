# Here are your Instructions
# Terminal 1
cd backend
conda activate banana-clean
uvicorn server:app --reload --port 8000
# Terminal 2
cd frontend
npm run start:react

