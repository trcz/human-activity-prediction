# human-activity-prediction
Multi-class prediction problem based on Human Activity Record from smartphones

**Dataset source:** http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions  
Dataset comes in preprocessed form, ready for importing  

**Instructions:**  
- download and extract dataset  
- place Test and Train in script directory  
- run activityNB.py  

I've used GaussianNB model and achieved 74.73% precision. Considering the fact GaussianNB supports multiclass problems I'm pretty disappointed with its accuracy.  
I will try another multiclass models to reach better results.
