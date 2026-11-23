# KeyNography API

## ✨ Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- Stegano, Pillow, OpenCV

## 🛠️ Setup

1. Clone this repository

2. Create a virtual environment
   ```powershell
   python -m venv .venv
   ```

3. Install dependencies
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

## 🏃‍♀️ Run the app

1. Open your terminal and run the API
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 7000
   ```

2. Open your browser and go to http://127.0.0.1:7000/docs
