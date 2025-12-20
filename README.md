# Fake News and Misinformation Detector

This project is an AI-based Fake News and Misinformation Detection system designed to identify misleading or false information from news articles and online content.

## Problem Statement
The rapid spread of fake news on social media and digital platforms can mislead users and create social harm. There is a need for an automated system that helps users verify information before sharing it.

## Proposed Solution
The system uses Machine Learning and Natural Language Processing (NLP) techniques to analyze text content and classify it as fake or real based on patterns, keywords, and context.

## Technologies Used
- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn / Pandas (planned)

## Future Scope
- Integration with browser extensions
- Real-time social media analysis
- Multilingual fake news detection
- 
# Fake News Detector

A Flask-based web application that detects fake news using Machine Learning (Logistic Regression with TF-IDF vectorization).

## Features

- **Web UI**: User-friendly interface to test news articles in real-time
- **REST API**: JSON API endpoint for programmatic access
- **ML Model**: Logistic Regression classifier trained on labeled data
- **TF-IDF Vectorization**: Uses bigram features for better accuracy
- **Real-time Predictions**: Instant classification with confidence scores
- **Confidence Metrics**: Shows prediction percentages for fake/real classification

## Project Structure

```
fake-news-detector/
├── fake_news_detector.py      # Main Flask application
├── ts_model.joblib            # Trained ML model (auto-generated)
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── dfl_env/                   # Virtual environment (excluded from git)
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

2. **Create a virtual environment**:
```bash
python -m venv dfl_env
```

3. **Activate the virtual environment**:

**Windows (PowerShell)**:
```powershell
dfl_env\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:
```cmd
dfl_env\Scripts\activate.bat
```

**macOS/Linux**:
```bash
source dfl_env/bin/activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install flask scikit-learn pandas joblib
```

## Usage

### Running the Application

```bash
python fake_news_detector.py
```

The app will start on **http://127.0.0.1:5000**

### Web Interface
- Open http://127.0.0.1:5000 in your browser
- Paste a news article or headline in the text area
- Click "Check" to get predictions

### REST API

**Endpoint**: `POST /api/predict`

**Request Example**:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Breaking: Scientists discovered cure for rare disease"}'
```

**Response Example**:
```json
{
  "label": "fake",
  "confidence": 95.5,
  "details": "fake: 95.50%\nreal: 4.50%"
}
```

## Model Information

- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (1-2 grams, max 4000 features)
- **Training Data**: 10 manually curated news samples
- **Test Accuracy**: ~67% (demo model - use larger dataset for production)
- **Model File**: `ts_model.joblib`

## Training Data

The model is trained on:
- 5 fake news samples (clickbait, health scams, false claims)
- 5 real news samples (government updates, research, factual reports)

For better accuracy in production:
- Use 1000+ labeled samples
- Include diverse news sources
- Apply text preprocessing (stemming, lemmatization)
- Use cross-validation for better evaluation

## Dependencies

- **flask**: Web framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **joblib**: Model serialization

See `requirements.txt` for versions.

## Development

### Modify the Model
Edit the `data` list in `fake_news_detector.py` to add/change training samples.

### Change Port
Modify the last line:
```python
app.run(debug=True, use_reloader=False, port=8080)  # Change port to 8080
```

### Disable Debug Mode (Production)
```python
app.run(debug=False, use_reloader=False)
```

## Deployment

For production deployment, use a WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 fake_news_detector:app
```

## Limitations

- Small training dataset (demo only)
- Limited to English text
- No preprocessing or stemming
- May not handle sarcasm or satire well
- Should not be used as sole source for news verification

## Future Improvements

- [ ] Integrate larger labeled dataset
- [ ] Add text preprocessing
- [ ] Implement multiple ML models (Random Forest, SVM, Neural Networks)
- [ ] Add feature importance visualization
- [ ] Database integration for logging predictions
- [ ] User authentication
- [ ] Multi-language support

## License

MIT License - Feel free to use and modify

## Author

Created as an AI for Trust & Safety Demo Project

## Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: This is a demonstration project. For real-world news verification, use established fact-checking services and combine multiple sources.

