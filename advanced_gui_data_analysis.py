import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import logging
from pathlib import Path

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, model_type: str):
        models = {
            'Regression': {
                'RandomForest': RandomForestRegressor(),
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor()
            },
            'Classification': {
                'RandomForest': RandomForestClassifier(),
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'DecisionTree': DecisionTreeClassifier()
            }
        }
        return models.get(model_type, {}).get(model_name)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
    
    def preprocess(self, data: pd.DataFrame) -> tuple:
        numeric_cols = data.select_dtypes(include=[np.number])
        X = self.imputer.fit_transform(numeric_cols.iloc[:, :-1])
        X = self.scaler.fit_transform(X)
        y = numeric_cols.iloc[:, -1]
        
        if np.isnan(y).any():
            y = pd.Series(y).fillna(y.mean())
        
        return X, y

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Analysis Tool")
        self.setup_logging()
        
        self.data: Optional[pd.DataFrame] = None
        self.model_type = tk.StringVar(value="Regression")
        self.selected_model = tk.StringVar(value="RandomForest")
        
        self.preprocessor = DataPreprocessor()
        self.setup_ui()
    
    def setup_logging(self):
        logging.basicConfig(
            filename='data_analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_ui(self):
        # Main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control frame for buttons and options
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Load file button
        self.load_btn = ttk.Button(
            control_frame,
            text="Load Dataset",
            command=self.load_file
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Model type selection
        model_type_frame = ttk.LabelFrame(control_frame, text="Model Type", padding="5")
        model_type_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            model_type_frame,
            text="Regression",
            variable=self.model_type,
            value="Regression"
        ).pack(side=tk.LEFT)
        
        ttk.Radiobutton(
            model_type_frame,
            text="Classification",
            variable=self.model_type,
            value="Classification"
        ).pack(side=tk.LEFT)
        
        ttk.Radiobutton(
            model_type_frame,
            text="Clustering",
            variable=self.model_type,
            value="Clustering"
        ).pack(side=tk.LEFT)
        
        # Model selection
        model_frame = ttk.LabelFrame(control_frame, text="Select Model", padding="5")
        model_frame.pack(side=tk.LEFT, padx=5)
        
        model_options = ["RandomForest", "LinearRegression", "DecisionTree", 
                        "LogisticRegression", "KMeans", "PCA"]
        self.model_dropdown = ttk.Combobox(
            model_frame,
            values=model_options,
            textvariable=self.selected_model,
            state="readonly"
        )
        self.model_dropdown.pack(side=tk.LEFT)
        
        # Analyze button
        self.analyze_btn = ttk.Button(
            control_frame,
            text="Analyze and Build Model",
            command=self.process_data,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Results area
        result_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to results area
        scroll = ttk.Scrollbar(result_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            result_frame,
            wrap=tk.WORD,
            yscrollcommand=scroll.set,
            font=("Helvetica", 10)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.result_text.yview)

    def load_file(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            if not file_path:
                return
            
            file_extension = Path(file_path).suffix
            self.data = pd.read_csv(file_path) if file_extension == '.csv' else pd.read_excel(file_path)
            
            self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.show_data_summary()
            self.analyze_btn.config(state=tk.NORMAL)
            
            logging.info(f"Successfully loaded dataset: {file_path}")
            
        except Exception as e:
            logging.error(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def show_data_summary(self):
        summary = f"""
        Dataset Summary:
        - Shape: {self.data.shape}
        - Missing Values: {self.data.isnull().sum().sum()}
        - Numeric Columns: {len(self.data.select_dtypes(include=[np.number]).columns)}
        
        Available Columns:
        {', '.join(self.data.columns)}
        """
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, summary)

    def process_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        try:
            X, y = self.preprocessor.preprocess(self.data)
            
            if self.model_type.get() == "Clustering":
                self.run_clustering(X)
            else:
                self.run_supervised(X, y)
                
        except Exception as e:
            logging.error(f"Error during data processing: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")

    def run_supervised(self, X, y):
        model = ModelFactory.get_model(
            self.selected_model.get(),
            self.model_type.get()
        )
        
        if model is None:
            messagebox.showerror("Error", "Invalid model selection")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Add cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        self.show_model_results(model, X_test, y_test, predictions, cv_scores)

    def show_model_results(self, model, X_test, y_test, predictions, cv_scores):
        if self.model_type.get() == "Regression":
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = model.score(X_test, y_test)
            
            results = f"""
            Model Evaluation Results:
            - Mean Squared Error (MSE): {mse:.4f}
            - Root Mean Squared Error (RMSE): {rmse:.4f}
            - R-squared (R2) Score: {r2:.4f}
            - Cross-validation scores (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
            """
            
            self.result_text.insert(tk.END, results)
            self.show_regression_plot(y_test, predictions)
        else:
            accuracy = accuracy_score(y_test, predictions)
            results = f"""
            Model Evaluation Results:
            - Accuracy: {accuracy:.4f}
            - Cross-validation scores (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
            
            Classification Report:
            {classification_report(y_test, predictions)}
            """
            
            self.result_text.insert(tk.END, results)
            self.show_classification_report(y_test, predictions)

    def show_regression_plot(self, y_test, predictions):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

    def show_classification_report(self, y_test, predictions):
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).iloc[:-1, :].T,
            annot=True,
            cmap='Blues'
        )
        plt.title('Classification Report')
        plt.show()

    def run_clustering(self, X):
        if self.selected_model.get() == "KMeans":
            model = KMeans(n_clusters=3, random_state=42)
            clusters = model.fit_predict(X)
            
            results = f"""
            K-Means Clustering Results:
            - Number of clusters: 3
            - Samples in each cluster:
            {pd.Series(clusters).value_counts().to_string()}
            """
            
            self.result_text.insert(tk.END, results)
            
            if X.shape[1] >= 2:
                plt.figure(figsize=(10, 6))
                plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
                plt.title('Cluster Visualization')
                plt.show()
                
        elif self.selected_model.get() == "PCA":
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X)
            
            explained_var = pca.explained_variance_ratio_
            results = f"""
            PCA Results:
            - Explained variance ratio (1st component): {explained_var[0]:.4f}
            - Explained variance ratio (2nd component): {explained_var[1]:.4f}
            """
            
            self.result_text.insert(tk.END, results)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA Results')
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.geometry("1000x600")
    root.mainloop()