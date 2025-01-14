# 🌟 **Sales Forecasting for Rossmann Pharmaceuticals**

## 📊 **Overview**
Welcome to the **Sales Forecasting** project for **Rossmann Pharmaceuticals**! This cutting-edge machine learning solution predicts store sales across various cities **up to six weeks in advance**. 

---

## 💡 **Business Need**
Rossmann Pharmaceuticals aims to enhance sales forecasting accuracy. Currently, store managers rely on intuition, leading to inconsistencies. With a data-driven approach, we provide a robust solution for forecasting sales, improving inventory management and resource allocation.

---

## 🔑 **Key Features**
- **Accurate Predictions**: this model considers critical factors such as:
  - **Promotions**
  - **Competition**
  - **Holidays**
  - **Seasonality**
  - **Locality**

---

## 📂 **Project Structure**
The project is meticulously organized for reproducibility and scalability across data processing, modeling, and visualization. Here’s a snapshot of the structure:

```
+---.github
|   \---workflows
|           blank.yml
| 
+---app
|   |---templates
|           index.html
|           result.html
|       .gitignore
|       app.py
|       model.py
|       preprocessing.py
|       requirements.txt 
+---.vscode
|       settings.json
|       
+---notebooks
|       __init__.ipynb
|       data_preprocessing.ipynb
|       eda_analysis.ipynb
|       README.md
|       sales_prediction_model.ipynb
|                         
+---scripts
|   |   __init__.py
|   |   data_preprocessing.py
|   |   data_processing.py
|   |   data_visualization.py
|   |   load_data.py
|   |   README.md
|   |   sales_model_pipeline.py  
+---src
|       README.md
|       __init__.py
|       
|---tests
        README.md
        __init__.py
|   .gitignore
|   README.md
|   requirements.txt
\  
```
## 🚀 **Installation**

1. To set up the project, follow these steps:

Clone the repository:

```
git clone https://github.com/Amen-Zelealem/Rossmann-Sales-Prediction
cd Rossmann-Sales-Prediction
```

2. Install the required packages:

```
pip install -r requirements.txt
```

## 📈 **Visualization**

### **Outlier Detection**

![Outlier](/screenshots/outlier.png)


### **Promo Distributions**

![promotionDistribution](/screenshots/promoDistribution.png)


### **Sales Behaviour Before, During and after Holidays**

![salesBehaviour](/screenshots/salesBehaviour.png)


###  🖥️ **FLASK APPLICATION SCREENSHOTS**

![predictionForm](/screenshots/predictionForm.png)


![predictionResult](/screenshots/predictionResult.png)

For more experience, you can refer to this site on [Rossmann Sales Prediction](https://flaskdeployment-83p9.onrender.com/).

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: 
```git checkout -b feature-branch ```  
3. Make your changes and commit: 
```git commit -m 'Add new feature' ```  
4. Push to the branch: 
```git push origin feature-branch```   
5. Open a pull request.
   
---Thank you for your interest in the **Rossmann Sales Forecasting project**! 
Let's make sales predictions smarter together! 
