# Satellite Image Interpolation using RIFE

## 📌 Overview
This project focuses on using **RIFE (Real-Time Intermediate Flow Estimation)**, a deep learning-based frame interpolation method, to enhance satellite imagery by generating intermediate frames. This improves temporal resolution, helping in better monitoring of weather patterns, environmental changes, and urban planning.

## 🚀 Features
- **Deep Learning-based Interpolation**: Uses RIFE to generate smooth transitions between satellite image frames.
- **Improved Temporal Resolution**: Enhances monitoring capabilities by interpolating missing frames.
- **High-Quality Image Processing**: Retains image details and minimizes artifacts.
- **Scalability**: Can be adapted to different satellite datasets and resolutions.
- **Efficient Implementation**: Optimized for real-time or near real-time processing.

## 🛠️ Technologies Used
- **Python**
- **TensorFlow / PyTorch**
- **RIFE Model**
- **OpenCV** (for image processing)
- **NumPy & Matplotlib** (for analysis and visualization)

## 📂 Dataset
The project works with time-series satellite imagery datasets, such as:
- Sentinel-2
- Landsat-8
- MODIS

## 📌 Installation & Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/shantanu0603/satellite-image-interpolation.git
   cd satellite-image-interpolation
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Model**
   ```bash
   python interpolate.py --input_folder images/
   ```

## 📊 Results & Impact
- Generated high-quality interpolated frames for satellite imagery.
- Enhanced temporal resolution, improving insights for environmental monitoring.
- Potential applications in **disaster response, climate change analysis, and land-use mapping**.

## 👨‍💻 Future Improvements
- Integrating AI-based denoising for sharper interpolations.
- Expanding to multi-spectral imagery.
- Deploying as an API/web service for real-time processing.

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.
