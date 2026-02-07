# Credit Card Fraud Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

##  About the Project

This project implements a comprehensive machine learning solution for credit card fraud detection, combining both supervised and unsupervised learning techniques. Using a dataset of 284,807 credit card transactions, the system identifies fraudulent transactions while addressing the challenges of highly imbalanced data (fraud rate: 0.172%).

The project demonstrates practical applications of:
- **Supervised Learning**: Logistic Regression for binary fraud classification
- **Unsupervised Learning**: K-Means clustering for transaction pattern segmentation
- **Hybrid Analysis**: Evaluating supervised model performance across unsupervised cluster segments

##  Goal

The primary objectives of this project are:

1. **Build a robust fraud detection model** capable of identifying fraudulent transactions with high recall to minimize financial losses
2. **Discover natural transaction patterns** using unsupervised clustering to understand risk segmentation
3. **Analyze model performance** across different transaction segments to identify strengths and weaknesses
4. **Handle class imbalance** effectively using appropriate techniques and evaluation metrics
5. **Provide actionable insights** for real-world fraud prevention systems

##  Why This Project?

Credit card fraud is a critical problem in the financial industry, with billions of dollars lost annually. Traditional rule-based systems struggle to adapt to evolving fraud patterns. This project addresses several key challenges:

- **Highly Imbalanced Data**: Only 0.172% of transactions are fraudulent, making standard classification approaches ineffective
- **Real-time Detection Needs**: The model must quickly identify fraud without disrupting legitimate transactions
- **Pattern Discovery**: Understanding different types of fraud patterns helps in developing targeted prevention strategies
- **Cost-Sensitive Classification**: False negatives (missed frauds) are more costly than false positives (blocked legitimate transactions)

This project demonstrates how machine learning can provide adaptive, data-driven solutions that continuously improve fraud detection capabilities.

## 🔬 Methodology

### 1. Data Collection
- **Dataset**: Credit Card Transactions Dataset (Kaggle)
- **Size**: 284,807 transactions
- **Features**: 30 features (28 PCA-transformed features V1-V28, Time, Amount, Class)
- **Target Variable**: Class (0 = Legitimate, 1 = Fraud)
- **Class Distribution**: 492 frauds (0.172%) vs 284,315 legitimate transactions (99.828%)

### 2. Exploratory Data Analysis (EDA)
- Statistical analysis of transaction distributions
- Fraud vs legitimate transaction comparison
- Feature correlation analysis
- Identification of time and amount patterns

### 3. Data Preprocessing
- **Feature Engineering**: 
  - Extracted `Hour` from `Time` feature for temporal analysis
  - Selected 30 features for modeling
- **Feature Scaling**: StandardScaler applied to normalize features
- **Train-Test Split**: 70% training (199,364 samples), 30% testing (85,443 samples)
- **Class Imbalance Handling**: Used `class_weight='balanced'` in Logistic Regression

### 4. Supervised Learning: Logistic Regression

**Model Configuration**:
```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear'
)
```

**Why Logistic Regression?**
- Interpretable coefficients for feature importance analysis
- Probabilistic predictions for risk scoring
- Efficient training on large datasets
- Well-suited for binary classification
- Built-in support for class imbalance through `class_weight`

### 5. Unsupervised Learning: K-Means Clustering

**Optimal Cluster Selection**:
- **Method**: Elbow method with silhouette score analysis
- **Sample Size**: 30,000 transactions (for computational efficiency)
- **K Range Tested**: 2 to 10 clusters
- **Optimal k**: 9 clusters (silhouette score: 0.0910)

**Final Model Configuration**:
```python
MiniBatchKMeans(
    n_clusters=9,
    random_state=42,
    n_init=10,
    batch_size=2000
)
```

**Risk Classification Framework**:
- **HIGH RISK**: Fraud rate > 0.5%
- **MEDIUM RISK**: Fraud rate 0.2% - 0.5%
- **LOW RISK**: Fraud rate < 0.2%

### 6. Hybrid Analysis
- Evaluated Logistic Regression performance across all 9 K-Means clusters
- Analyzed how supervised model handles different transaction patterns
- Identified clusters where model excels vs struggles

##  Evaluation Metrics

### Primary Metrics
Given the severe class imbalance, we prioritized metrics that focus on minority class performance:

1. **Recall (Sensitivity)**: Proportion of actual frauds correctly identified
   - *Most critical metric* - missing fraud is costly
   
2. **Precision**: Proportion of predicted frauds that are actually fraudulent
   - Balances false positive rate
   
3. **F1-Score**: Harmonic mean of precision and recall
   - Overall model effectiveness measure
   
4. **ROC-AUC**: Area under the Receiver Operating Characteristic curve
   - Model's ability to discriminate between classes
   
5. **Confusion Matrix**: Detailed breakdown of prediction outcomes

### Why These Metrics?

**Accuracy is misleading**: A model predicting all transactions as legitimate achieves 99.828% accuracy but catches zero frauds. Therefore, we focus on:

- **Recall > Precision**: Better to flag legitimate transactions for review than miss frauds
- **ROC-AUC**: Evaluates performance across all classification thresholds
- **Cluster-specific metrics**: Reveals model behavior across different transaction patterns

##  Challenges

### 1. Extreme Class Imbalance (0.172% fraud rate)
- **Problem**: Standard models bias toward majority class
- **Solution**: Used `class_weight='balanced'` and specialized evaluation metrics

### 2. Anonymized Features (V1-V28)
- **Problem**: Cannot interpret PCA-transformed features directly
- **Solution**: Focused on pattern recognition and relative importance rather than feature semantics

### 3. Computational Efficiency
- **Problem**: Training K-Means on 199,364 samples is computationally expensive
- **Solution**: Used 30,000-sample subset for k optimization, then trained on full dataset with MiniBatchKMeans

### 4. Low Precision in Low-Risk Clusters
- **Problem**: Model over-predicts fraud in clusters with very low fraud rates (<0.1%)
- **Insight**: Inherent challenge when base rate is extremely low; prioritizes recall over precision

### 5. Varying Performance Across Clusters
- **Problem**: Model performance differs significantly across transaction segments
- **Opportunity**: Suggests potential for cluster-specific optimization strategies

##  Results and Insights

### Overall Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.971** |
| **Recall (Fraud)** | **0.872** |
| **Precision (Fraud)** | **0.065** |
| **F1-Score (Fraud)** | **0.121** |
| **Accuracy** | **97.81%** |

**Detailed Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Legitimate** | 0.9998 | 0.9783 | 0.9889 | 85,295 |
| **Fraud** | 0.0651 | 0.8716 | 0.1212 | 148 |

**Confusion Matrix**:
- **True Negatives**: 83,443 (correctly identified legitimate transactions)
- **False Positives**: 1,852 (legitimate flagged as fraud)
- **False Negatives**: ~19 (frauds missed)
- **True Positives**: ~129 (frauds caught)

**Interpretation**:
- **Excellent ROC-AUC (0.971)**: Model strongly discriminates between fraud and legitimate transactions across all thresholds
- **Strong Recall (87.2%)**: Captures majority of fraudulent transactions - critical for minimizing financial losses
- **Low Precision (6.5%)**: High false positive rate means ~15 legitimate transactions flagged for every actual fraud detected
- **Trade-off Justification**: Missing a fraud (cost: full transaction amount) is far more expensive than reviewing a legitimate transaction (cost: customer inconvenience), making the recall-focused approach appropriate

### K-Means Clustering Results

**9 Clusters Identified**:

| Cluster | Size | Fraud Rate | Risk Level | Key Characteristics |
|---------|------|------------|------------|---------------------|
| **0** | 14.5% | 0.132% | LOW | Moderate amounts ($99), standard hours |
| **1** | 22.0% | 0.055% | LOW | Standard amounts ($81), standard hours |
| **2** | 15.6% | 0.138% | LOW | Lower amounts ($64), standard hours |
| **3** | 20.5% | 0.042% | LOW | Higher amounts ($137), standard hours |
| **4** | 0.8% | **3.307%** | **HIGH** | **Very high amounts ($393)** |
| **5** | 4.6% | 0.065% | LOW | Moderate-high amounts ($125) |
| **6** | 7.0% | 0.475% | MEDIUM | Low amounts ($60), **unusual hours (3.9h)** |
| **7** | 0.9% | **0.969%** | **HIGH** | **Very low amounts ($17)** |
| **8** | 14.1% | 0.291% | MEDIUM | Low amounts ($35), late hours (16.7h) |

**Key Patterns**:
1. **High-value fraud** (Cluster 4): Small cluster with exceptionally high fraud rate (3.307%)
2. **Low-value fraud** (Cluster 7): Another high-risk cluster with very small transaction amounts
3. **Temporal patterns** (Cluster 6): Unusual transaction hours correlate with elevated fraud risk
4. **Majority low-risk**: 77.2% of transactions fall into low-risk clusters

##  Model Performance by Cluster

### Exceptional Performance (High-Risk Clusters)

**Cluster 4** (Highest Risk - 3.355% fraud rate):
-  **Perfect Recall**: 1.0000 (catches all frauds)
-  **Precision**: 0.2593 (reasonable given high fraud prevalence)
-  **Accuracy**: 90.42%

**Cluster 7** (High Risk - 1.292% fraud rate):
-  **Perfect Recall**: 1.0000
-  **Strong Precision**: 0.6667 (best balance achieved)
-  **Accuracy**: 99.35%

### Strong Performance (Medium-Risk Clusters)

**Cluster 6** (Medium Risk - 0.331% fraud rate):
-  **Recall**: 0.9500 (catches 95% of frauds)
-  **Precision**: 0.0864
-  **Accuracy**: 96.66%

**Cluster 8** (Medium Risk - 0.318% fraud rate):
-  **Recall**: 0.8947 (catches ~90% of frauds)
-  **Precision**: 0.0427
-  **Accuracy**: 93.58%

### Excellent Balance (Low-Risk Cluster)

**Cluster 2** (Low Risk - 0.164% fraud rate):
-  **Recall**: 0.9545 (exceptional for low-risk cluster)
-  **Precision**: 0.2333 (best among low-risk clusters)
-  **Accuracy**: 99.48%

### Challenging Performance (Ultra-Low Risk Clusters)

**Cluster 3** (0.040% fraud rate):
-  **Recall**: 0.2857 (worst performance)
-  **Precision**: 0.0128 (many false positives)
-  **Accuracy**: 99.09%

**Clusters 0, 1, 5** (0.05-0.13% fraud rates):
-  **Recall**: 0.69-1.00 (acceptable to excellent)
-  **Precision**: 0.02-0.04 (very low)
-  **Accuracy**: 97-99%

##  Key Insights

### 1. Inverse Relationship: Fraud Rate vs Precision
- **High fraud rate clusters** (4, 7): Higher precision (25-67%) with perfect recall
- **Low fraud rate clusters** (0-3, 5): Very low precision (1-4%) but acceptable recall
- **Implication**: Model accuracy increases when fraud is more prevalent in the segment

### 2. Model Prioritizes Recall Over Precision
- Across all clusters, recall ranges from 28.57% to 100%
- Five clusters achieve ≥95% recall
- Model design successfully minimizes false negatives (missed frauds)
- Trade-off: Higher false positive rate, especially in low-risk segments

### 3. Clustering Validates Transaction Segmentation
- Performance variation (1.28%-66.67% precision) proves clusters represent distinct patterns
- High-risk clusters have identifiable characteristics (extreme amounts, unusual hours)
- Unsupervised clustering successfully segments data without fraud labels

### 4. Pattern-Specific Detection Capabilities
- **Best performance**: High-value fraud (Cluster 4) and low-value fraud (Cluster 7)
- **Strong performance**: Unusual timing patterns (Cluster 6)
- **Weakest performance**: Ultra-low risk, standard transactions (Cluster 3)

### 5. Time-of-Day Matters
- Cluster 6 (avg hour: 3.9h) shows elevated fraud rate (0.475%)
- Late-hour transactions (Cluster 8, avg hour: 16.7h) also show increased risk (0.291%)
- Standard business hours (Clusters 0-3, avg ~15h) have lower fraud rates

##  Recommendations

### For Production Deployment

1. **Risk-Adjusted Decision Thresholds**
   - **High-risk clusters (4, 7)**: Use standard threshold (0.5) given strong precision
   - **Low-risk clusters (0-3, 5)**: Increase threshold (0.7-0.8) to reduce false positives
   - **Medium-risk clusters (6, 8)**: Use moderate threshold (0.6)

2. **Tiered Review Process**
   - **Cluster 4 & 7 flags**: Immediate manual review or auto-block
   - **Cluster 6 & 8 flags**: Secondary automated screening before review
   - **Other clusters**: Lower priority review queue

3. **Temporal Monitoring**
   - Enhanced scrutiny for transactions during unusual hours (midnight-6am)
   - Implement time-of-day risk multipliers

4. **Amount-Based Rules**
   - Flag high-value transactions (>$300) for additional verification
   - Flag very low-value transactions (<$20) from new/unusual locations

### For Model Improvement

1. **Cluster-Specific Sub-Models**
   - Train specialized models for Cluster 3 (worst performer)
   - Develop high-risk cluster models with different thresholds

2. **Additional Feature Engineering**
   - Derive velocity features (transactions per hour/day)
   - Calculate deviation from user's historical patterns
   - Include merchant category codes if available

3. **Ensemble Approaches**
   - Combine Logistic Regression with tree-based models (Random Forest, XGBoost)
   - Use stacking with cluster assignments as meta-features

4. **Advanced Imbalance Techniques**
   - Implement SMOTE (Synthetic Minority Over-sampling)
   - Try cost-sensitive learning with custom loss functions
   - Experiment with anomaly detection algorithms

### For Business Strategy

1. **Customer Communication**
   - Prepare clear messaging for false positive scenarios
   - Implement one-click verification for flagged transactions
   - Educate customers on fraud patterns

2. **Dynamic Threshold Adjustment**
   - Monitor false positive rates by cluster
   - Adjust thresholds based on seasonal patterns
   - A/B test different threshold strategies

3. **Fraud Pattern Monitoring**
   - Track cluster distribution shifts over time
   - Alert when new high-risk patterns emerge
   - Retrain models quarterly with updated data

##  Conclusion

This project successfully demonstrates a hybrid machine learning approach to credit card fraud detection:

**Achievements**:
 Built a Logistic Regression model with 97.1% ROC-AUC and 87.2% recall
 Identified 9 distinct transaction patterns through K-Means clustering
 Discovered 2 high-risk clusters with fraud rates 5-19x the baseline
 Achieved perfect recall (100%) on high-risk transaction segments
 Provided actionable insights for production fraud detection systems

**Key Takeaways**:
1. **Unsupervised clustering reveals meaningful risk segmentation** even without fraud labels
2. **Model performance varies significantly by transaction pattern**, suggesting cluster-specific strategies
3. **Prioritizing recall (87.2%) over precision (6.5%)** is appropriate for fraud detection given cost asymmetry
4. **Hybrid approaches** (supervised + unsupervised) provide richer insights than either alone

**Real-World Impact**:
- **87.2% of frauds detected**, preventing significant financial losses
- **1,852 false positives** out of 85,295 legitimate transactions (2.17% false positive rate)
- **Risk segmentation** enables targeted monitoring and resource allocation
- **Pattern insights** inform fraud prevention strategies and customer education
- **Scalable approach** using MiniBatchKMeans for production deployment

The combination of supervised and unsupervised techniques creates a robust, interpretable fraud detection system suitable for production deployment with appropriate threshold tuning and monitoring.

##  Future Work

### Short-Term Enhancements
1. **Hyperparameter Optimization**
   - Grid search for optimal Logistic Regression parameters
   - Tune class weights for better precision-recall balance
   - Optimize K-Means initialization and distance metrics

2. **Model Comparison**
   - Benchmark against Random Forest, XGBoost, LightGBM
   - Compare with anomaly detection algorithms (Isolation Forest, One-Class SVM)
   - Evaluate neural network approaches (Autoencoders)

3. **Advanced Evaluation**
   - Implement time-series cross-validation
   - Calculate cost-based metrics ($ saved vs $ lost to false positives)
   - Analyze model performance degradation over time

### Medium-Term Developments
4. **Feature Engineering**
   - Engineer transaction velocity features
   - Create merchant risk scores
   - Develop behavioral deviation metrics

5. **Ensemble Methods**
   - Implement weighted voting ensemble
   - Use cluster assignments as meta-features in stacking
   - Combine multiple model predictions with learned weights

6. **Explainability**
   - Implement SHAP values for individual predictions
   - Create dashboard visualizing fraud risk factors
   - Develop interpretable rule extraction from models

### Long-Term Research
7. **Deep Learning**
   - Recurrent Neural Networks for sequential transaction patterns
   - Autoencoders for anomaly detection
   - Graph Neural Networks for transaction networks

8. **Real-Time System**
   - Implement streaming prediction pipeline
   - Develop online learning for model updates
   - Create A/B testing framework for threshold optimization

9. **Advanced Clustering**
   - Explore DBSCAN for density-based clustering
   - Implement hierarchical clustering for pattern taxonomy
   - Try Gaussian Mixture Models for probabilistic clustering

10. **Causal Analysis**
    - Investigate causal relationships in fraud patterns
    - Develop counterfactual explanations
    - Identify fraud prevention intervention points

---

##  Technologies Used

- **Language**: Python 3.8+
- **Core Libraries**: 
  - scikit-learn (ML models and metrics)
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib & seaborn (visualization)
- **Algorithms**: 
  - Logistic Regression (supervised learning)
  - MiniBatch K-Means (unsupervised learning)



##  Dataset

The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, containing transactions made by European cardholders in September 2013.

##  Author

**Korede**
- MSc Data Science Student, York St John University - London School
- [LinkedIn](https://www.linkedin.com/in/folarinkorede/)
- [GitHub](https://github.com/korede-folarin)

##  License

This project is licensed under the MIT License 

##  Acknowledgments

- Dataset provided by Machine Learning Group - ULB (Université Libre de Bruxelles)
- Inspired by real-world challenges in fraud detection systems
- Thanks to the scikit-learn community for excellent documentation

---

**If you found this project helpful, please consider giving it a star!**
