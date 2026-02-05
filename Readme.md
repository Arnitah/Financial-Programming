# Financial Programming

Welcome to the **Financial Programming** repository! This repository focuses on analyzing financial datasets and creating practical financial solutions such as loan prediction models and stock pricing dashboards.

## What's Inside?
The repository contains two key applications:
1. **Loan Prediction Analysis**:
   - Implements machine learning techniques to predict loan approvals based on financial datasets.
   - Includes data preprocessing, feature engineering, and model evaluation.
2. **Stock Pricing Dashboard**:
   - Interactive stock pricing dashboard built using **Streamlit**.
   - Retrieves real-time stock price data using Yahoo Finance API for visualization and analysis.

---

## Key Features
- **Loan Prediction**:
  - Data cleaning and preprocessing scripts.
  - Statistical analysis and visualization of trends in loan datasets.
  - Prediction models for loan approval using supervised learning techniques.
  
- **Yahoo Stock Pricing Dashboard**:
  - Intuitive interface to display stock prices and trends.
  - Integration with Yahoo Finance for live updates.
  - Visualizations of daily, weekly, and monthly pricing performance.

---

## File Structure
```
├── datasets/               # Contains financial datasets
├── loan_prediction/        # Scripts and notebooks for loan prediction analysis
├── stock_dashboard/        # Streamlit code for stock pricing dashboard
├── requirements.txt        # Python dependencies
├── README.md               # Repository information
```

---

## Getting Started

### Prerequisites
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Running the Loan Prediction Analysis
Navigate to the `loan_prediction` folder and run the scripts or notebooks for loan prediction insights:
```bash
cd loan_prediction
# Open the main analysis notebook
jupyter notebook Loan_Prediction_Analysis.ipynb
```

### Running the Stock Pricing Dashboard
1. Navigate to the `stock_dashboard` folder.
2. Launch the Streamlit dashboard:
```bash
cd stock_dashboard
streamlit run app.py
```
3. Open your browser to the provided local URL to interact with the dashboard.

---

## Tools and Libraries Used
- **Data Analysis**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Web Application**: Streamlit
- **Data Sources**: Yahoo Finance API

---

## Possible Enhancements
- Add predictive analytics for stock prices using advanced financial models.
- Implement loan risk categorization for better insights.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- Contributors and community members for their valuable feedback!

---

Feel free to contribute by submitting issues or pull requests. Happy coding!
