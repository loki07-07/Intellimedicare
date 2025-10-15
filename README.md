# Intellimedicare 🏥🤖

An intelligent healthcare platform that leverages AI and machine learning to provide comprehensive medical analysis and assistance across multiple domains.

## 🌟 Features

### 🧠 Brain Tumor Detection
- **Deep Learning Model**: CNN-based brain tumor classification
- **Multi-class Detection**: Identifies Glioma, Meningioma, Pituitary tumors, and No tumor cases
- **MRI Analysis**: Processes brain MRI scans for accurate diagnosis
- **Interactive Web Interface**: Easy-to-use upload and analysis system

### 🫁 Pneumonia Detection
- **X-ray Analysis**: Automated pneumonia detection from chest X-rays
- **PyTorch Implementation**: Advanced deep learning model for medical imaging
- **Quick Diagnosis**: Rapid analysis for healthcare professionals

### 🦴 Knee Injury Assessment
- **Multi-view Analysis**: Supports Axial, Coronal, and Sagittal views
- **CNN + Logistic Regression**: Hybrid approach for comprehensive analysis
- **Injury Classification**: Detects ACL tears, Meniscus damage, and general abnormalities
- **Batch Processing**: Handle multiple scans simultaneously

### 🩺 Skin Cancer Detection
- **Dermatological AI**: Automated skin lesion analysis
- **Image Classification**: Identifies potentially malignant skin conditions
- **Early Detection**: Assists in early-stage cancer identification

### 🧬 Breast Cancer Analysis
- **Comprehensive Screening**: Advanced breast cancer detection algorithms
- **Medical Imaging**: Supports various imaging modalities

### 📝 Handwritten Prescription Recognition
- **OCR Technology**: Optical Character Recognition for handwritten prescriptions
- **Medicine Database**: Extensive database of medicines with uses and side effects
- **Drug Information**: Detailed information about medications including dosage and interactions
- **Prescription Processing**: Converts handwritten prescriptions to digital format

### 🤖 Medical Chatbot
- **AI-Powered Assistance**: Intelligent medical query responses
- **Medicine Extraction**: Automatically extracts medicine information from queries
- **Healthcare Guidance**: Provides medical information and guidance

### 📊 Blood Report Analysis
- **Lab Report Processing**: Automated analysis of blood test results
- **Health Insights**: Provides insights based on blood parameters
- **Report Generation**: Creates comprehensive health reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/loki07-07/Intellimedicare.git
   cd Intellimedicare
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Files**
   - Due to GitHub file size limitations, model files need to be downloaded separately
   - Contact the repository owner for access to trained models
   - Place model files in their respective directories:
     - `Brain_tumor/model.h5`
     - `Pneumonia_Detection/pneumonia.pth`
     - `Skin_cancer/skincancer_model.pt`
     - `Knee_injury/cnn/` and `Knee_injury/lr/` directories

## 📁 Project Structure

```
Intellimedicare/
├── Brain_tumor/                 # Brain tumor detection module
│   ├── app.py                  # Flask application
│   ├── templates/              # HTML templates
│   ├── static/                 # CSS and static files
│   └── *.ipynb                 # Jupyter notebooks for training
├── Pneumonia_Detection/         # Pneumonia detection module
│   ├── app.py                  # Flask application
│   └── templates/              # HTML templates
├── Knee_injury/                # Knee injury assessment module
│   ├── app.py                  # Flask application
│   ├── model.py                # Model definitions
│   ├── utils.py                # Utility functions
│   └── templates/              # HTML templates
├── Skin_cancer/                # Skin cancer detection module
│   ├── app.py                  # Flask application
│   └── templates/              # HTML templates
├── Handwritten_recognition/     # Prescription OCR module
│   ├── app.py                  # Flask application
│   ├── ocr.py                  # OCR processing
│   ├── chat.py                 # Chat functionality
│   └── templates/              # HTML templates
├── chatbot/                    # Medical chatbot module
│   ├── app.py                  # Flask application
│   ├── extract_medicines.py    # Medicine extraction logic
│   └── Templates/              # HTML templates
├── Website/                    # Main website and user management
│   ├── app.py                  # Main Flask application
│   ├── templates/              # HTML templates
│   └── static/                 # CSS and static files
└── report_analysis/            # Blood report analysis module
    ├── app.py                  # Flask application
    └── templates/              # HTML templates
```

## 🔧 Usage

### Running Individual Modules

Each module can be run independently:

```bash
# Brain Tumor Detection
cd Brain_tumor
python app.py

# Pneumonia Detection
cd Pneumonia_Detection
python app.py

# Knee Injury Assessment
cd Knee_injury
python app.py

# Handwritten Prescription Recognition
cd Handwritten_recognition
python app.py
```

### Main Website
```bash
cd Website
python app.py
```

The application will be available at `http://localhost:5000`

## 🧪 Model Training

The repository includes Jupyter notebooks for training custom models:

- `Brain_tumor/brain_tumour_detection_using_deep_learning.ipynb`
- `Brain_tumor/brain-tumor-mri-classification-tensorflow-cnn.ipynb`
- `Pneumonia_Detection/pneumonia-detection.ipynb`
- `Handwritten_recognition/module_1.ipynb`

## 📊 Datasets

The models are trained on various medical datasets:
- Brain MRI images for tumor detection
- Chest X-rays for pneumonia detection
- Knee MRI scans for injury assessment
- Dermatoscopic images for skin cancer detection
- Handwritten prescription images for OCR

*Note: Due to privacy and size constraints, datasets are not included in this repository.*

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 👨‍💻 Authors

**Lokesh Dhanaraj**
- GitHub: [@loki07-07](https://github.com/loki07-07)
- Email: dhanarajlokesh000@gmail.com

**Aishwarya Shinde**
- GitHub : [@aishwaryashinde26](https://github.com/aishwaryashinde26)
- Email : aishwaryashinde7390@gmail.com

**Prince Ransom AC**
- Github : [@PrinceRansom7](https://github.com/PrinceRansom7)
- Email :princeransom7@gmail.com

## 🙏 Acknowledgments

- TensorFlow and PyTorch communities for excellent deep learning frameworks
- Flask framework for web application development
- OpenCV for image processing capabilities
- The medical imaging research community for datasets and methodologies

## 📞 Support

If you have any questions or need support, please:
1. Check the [Issues](https://github.com/loki07-07/Intellimedicare/issues) page
2. Create a new issue if your question isn't already addressed
3. Contact the author directly

---

*Made with ❤️ for advancing healthcare through AI*
