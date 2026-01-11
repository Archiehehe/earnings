# 📅 Earnings Radar

Upload your stock portfolio and get a calendar of upcoming earnings —  
with **context and historical market reactions** for the previous 4 quarters.

Live - earningsradar.streamlit.app

## 🚀 Overview

**Earnings Radar** is a lightweight finance tool that helps investors and traders
track earnings events for their portfolio in one place.

You can:
- Upload your portfolio
- View a **future earnings calendar**
- See **historical price reactions** around earnings
- Explore results interactively in a simple UI

Built with **Python** and designed for quick, practical earnings analysis.

## ✨ Features

- 📆 Upcoming earnings calendar for your portfolio
- 📊 Historical earnings reaction context (last 4 quarters)
- 📁 Upload your own portfolio file
- ⚡ Fast and simple to run locally
- 📈 Investor-focused insights without noise

📊 Output

The app provides:

A calendar of upcoming earnings dates

Historical earnings dates for the last 4 quarters

Market reaction context around each earnings event

Interactive tables and visuals inside the app

🧠 How It Works

Reads portfolio tickers from the uploaded CSV

Fetches earnings dates and historical price data

Computes price reactions around earnings

Displays results in a clean, interactive interface

🧪 Use Cases

Track earnings risk for your portfolio

Prepare for upcoming earnings weeks

Review how stocks historically react to earnings

Lightweight earnings research without heavy platforms

## 🛠️ Installation

**Requirements**
- Python 3.8+

Clone the repository:

```bash
git clone https://github.com/Archiehehe/earnings.git
cd earnings

##Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt


▶️ Running the App

Run the application with:

streamlit run app.py



