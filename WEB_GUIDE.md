# Web Interface Guide

## Overview

The IDS ML project now includes a complete web interface inspired by [RSK World](https://rskworld.in/contact.php), providing an easy-to-use interface for the Intrusion Detection System.

## Features

### Web Pages

1. **Home Page** (`index.html`)
   - Project overview and hero section
   - Key features showcase
   - Technologies used
   - Statistics and metrics

2. **About Page** (`about.html`)
   - Project overview
   - Team information (Molla Samser & Rima Khatun)
   - Technology stack details

3. **Demo Page** (`demo.html`)
   - Upload CSV files for analysis
   - Real-time attack detection
   - Model selection (Random Forest, SVM, Neural Network)
   - Results visualization

4. **Documentation Page** (`documentation.html`)
   - Installation instructions
   - Usage examples
   - API reference
   - Dataset format guide

5. **Contact Page** (`contact.html`)
   - Contact information
   - General inquiry form
   - Content removal request form
   - Statistics display

6. **Legal Pages**
   - Privacy Policy (`privacy.html`)
   - Terms & Conditions (`terms.html`)
   - Disclaimer (`disclaimer.html`)

## Running the Web Application

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (First Time)

Before using the web interface, you need to train the models:

```bash
python scripts/train.py
```

This will create the necessary model files in the `models/` directory.

### 3. Start the Web Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 4. Access the Web Interface

Open your browser and navigate to:
- Home: `http://localhost:5000/`
- Demo: `http://localhost:5000/demo.html`
- Contact: `http://localhost:5000/contact.html`
- Documentation: `http://localhost:5000/documentation.html`

## API Endpoints

### POST /api/predict
Upload a CSV file and get predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - `file`: CSV file (required)
  - `model`: Model name (random_forest, svm, neural_network)

**Response:**
```json
{
  "success": true,
  "total_samples": 100,
  "predictions": {
    "normal": 80,
    "dos": 15,
    "probe": 5
  },
  "accuracy": 95.5,
  "model": "random_forest"
}
```

### GET /api/models
List available trained models.

**Response:**
```json
{
  "available_models": ["random_forest", "svm", "neural_network"],
  "total": 3
}
```

### GET /api/stats
Get system statistics.

**Response:**
```json
{
  "models_loaded": 3,
  "available_models": ["random_forest", "svm", "neural_network"],
  "preprocessor_loaded": true
}
```

## Web Interface Structure

```
web/
├── index.html          # Home page
├── about.html          # About page
├── demo.html           # Demo/upload page
├── documentation.html  # Documentation
├── contact.html        # Contact page
├── privacy.html        # Privacy policy
├── terms.html          # Terms & conditions
├── disclaimer.html     # Disclaimer
├── css/
│   └── style.css       # Main stylesheet
└── js/
    ├── main.js         # Main JavaScript
    ├── contact.js      # Contact form handling
    └── demo.js         # Demo page functionality
```

## Customization

### Styling
Edit `web/css/style.css` to customize the appearance. The color scheme uses CSS variables:
- `--primary-color`: Main brand color (red)
- `--secondary-color`: Secondary color (blue)
- `--dark-color`: Dark text/background
- `--light-color`: Light background

### Content
All HTML pages can be edited directly in the `web/` directory.

### Contact Information
Update contact details in:
- `web/contact.html`
- `web/index.html` (footer)
- All other pages (footer sections)

## Features Inspired by RSK World

The web interface includes features from the RSK World contact page:

1. **Top Bar**: Contact information and social links
2. **Sticky Header**: Navigation that stays at the top when scrolling
3. **Contact Forms**: General contact and content removal forms
4. **Statistics Display**: Visitor and request statistics
5. **Responsive Design**: Mobile-friendly layout
6. **Team Information**: Founder and team member details
7. **Footer**: Comprehensive footer with links and contact info

## Browser Compatibility

The web interface is compatible with:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Security Notes

For production deployment:
1. Add authentication to `/api/train` endpoint
2. Implement rate limiting
3. Add CSRF protection
4. Use HTTPS
5. Validate and sanitize all inputs
6. Implement file size limits
7. Add logging and monitoring

## Troubleshooting

### Models Not Loading
- Ensure models are trained: `python scripts/train.py`
- Check that model files exist in `models/` directory
- Verify preprocessor is saved: `models/preprocessor.pkl`

### File Upload Issues
- Check file size (max 10MB)
- Ensure file is CSV format
- Verify file has correct column structure

### API Errors
- Check Flask console for error messages
- Verify models are loaded on startup
- Ensure data format matches expected structure

## Support

For issues or questions:
- Email: help@rskworld.in
- Phone: +91 93305 39277
- Visit: [Contact Page](contact.html)

