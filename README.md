# 📊 Data Description

This project uses the **Naval Air Training and Operating Procedures Standardization (NATOPS)** dataset. It focuses on body-hand gesture recognition during aircraft handling on the deck of an aircraft carrier.

## 👋 Gestures and Classes

- 6 of 24 possible gestures are included in this dataset.
- **Six gesture classes**:
  - I have command  
  - All clear  
  - Not clear  
  - Spread wings  
  - Fold wings  
  - Lock wings

## 📡 Sensor Data

- Data is collected using sensors placed on:
  - Hands  
  - Elbows  
  - Wrists  
  - Thumbs  
- Each joint provides 3D coordinates (x, y, z), resulting in **24 dimensions** total.

---

# 🛠️ Tasks

### 1. Read the Data
- Each `.arff` file in the `NATOPS.zip` folder represents one of the 24 dimensions.
- Files contain all samples (rows) and all time steps (columns).

### 2. Combine Samples into One Table
- Merge all 24 dimensions into one large table of shape **(time steps × features)**.
- Add a `sid` column to identify each sample.
- Output the result as a **CSV file**.

### 3. Preserve Train/Test Split
- The data is already split into training and testing sets.
- Add a `split` column to indicate whether a row is from the train or test set.

### 4. (Optional) Window Features
- Each sample contains **51 time steps**.
- You may apply windowing (e.g., percentage tectonic as described in the related paper) to improve model performance. This step is optional and experimental.

### 5. Final Output
- Save the final processed data as a `.csv` file.
- Display the first **5 rows** of the final table along with column names in your script output.

---
