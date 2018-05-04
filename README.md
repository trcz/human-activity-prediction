# human-activity-prediction
Multi-class prediction problm based on Human Activity Record from smartphones

**Dataset source:** http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions  
Dataset comes in preprocessed form, ready for importing  
All data comes normalized in range [-1; 1]

**Instructions:**  
- download and extract dataset  
- place Test and Train in script directory  
- run activity_prediction.py  

I've used GaussianNB model and achieved 74.73% precision. Considering the fact GaussianNB supports multiclass problems I'm pretty disappointed with its accuracy. Repeatable features could be the problem.  

I've chosen LinearSVC as another model in its One vs Rest version because OvR is good approach for multi-class data with repeatable features and achieved satisfying 94,62% accuracy.
