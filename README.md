# Water Quality Predictor — Production MLOps Project

Predict whether a water sample is **safe to drink** (potable) based on its chemical composition, using a full MLOps pipeline with MongoDB, EvidentlyAI drift monitoring, Docker, GitHub Actions CI/CD, and AWS ECR + EC2 deployment.

---

## Dataset

- **Source**: [Kaggle — Water Quality Dataset](https://www.kaggle.com/datasets/mssmartypants/water-quality)
- **Records**: 3,276 water samples
- **Target**: `Potability` (1 = Safe, 0 = Not Safe)
- **Features**: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Storage | MongoDB Atlas |
| ML Model | XGBoost Classifier |
| Imbalance | SMOTETomek (imblearn) |
| Drift Monitoring | EvidentlyAI |
| Web App | Flask |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS ECR + EC2 |
| Model Registry | AWS S3 |

---

## Project Structure

```
water_quality_predictor/
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD: lint → build → push ECR → deploy EC2
├── config/
│   ├── schema.yaml                # Column names, types, target
│   └── model.yaml                 # XGBoost hyperparameters
├── notebook/
│   └── EDA.ipynb                  # Exploratory data analysis
├── static/css/
│   └── style.css
├── templates/
│   └── index.html                 # Flask prediction UI
├── water_quality/
│   ├── components/
│   │   ├── data_ingestion.py      # MongoDB → CSV
│   │   ├── data_validation.py     # Schema checks + EvidentlyAI drift
│   │   ├── data_transformation.py # Imputer + RobustScaler + SMOTETomek
│   │   ├── model_trainer.py       # XGBoost training + evaluation
│   │   ├── model_evaluation.py    # Compare vs production model on S3
│   │   └── model_pusher.py        # Upload to S3 + local saved_models/
│   ├── constants/__init__.py      # All project-wide constants
│   ├── entity/
│   │   ├── config_entity.py       # Dataclass configs per pipeline stage
│   │   └── artifact_entity.py     # Dataclass outputs per pipeline stage
│   ├── exception/__init__.py      # Custom exception with file + line info
│   ├── logger/__init__.py         # Timestamped file logger
│   ├── pipeline/
│   │   ├── training_pipeline.py   # Orchestrates all 6 components
│   │   └── prediction_pipeline.py # Loads model, accepts form input
│   └── utils/
│       └── main_utils.py          # save/load object, YAML, numpy helpers
├── app.py                         # Flask app (/, /predict, /train)
├── demo.py                        # Run training pipeline standalone
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── setup.py
└── template.py                    # Scaffold project files
```

---

## Pipeline Stages

```
MongoDB Atlas
     │
     ▼
1. Data Ingestion      → Pull data, train/test split (80/20), save to CSV
     │
     ▼
2. Data Validation     → Schema check + EvidentlyAI drift report (YAML)
     │
     ▼
3. Data Transformation → Median imputation + RobustScaler + SMOTETomek
     │
     ▼
4. Model Trainer       → XGBoost (F1, Precision, Recall, ROC-AUC logged)
     │
     ▼
5. Model Evaluation    → Compare new model vs production model on S3
     │
     ▼
6. Model Pusher        → Upload to S3 if improved; serve locally via Flask
```

---

## Local Setup

### 1. Clone & create environment

```bash
git clone https://github.com/your-username/water-quality-predictor.git
cd water-quality-predictor

conda create -n waterquality python=3.10 -y
conda activate waterquality

pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_DEFAULT_REGION=us-east-1
```

### 3. Upload dataset to MongoDB

```python
import pandas as pd
from pymongo import MongoClient

client = MongoClient("your-mongodb-url")
df = pd.read_csv("notebook/water_potability.csv")
db = client["water_quality"]
db["water_data"].insert_many(df.to_dict("records"))
print("Uploaded:", db["water_data"].count_documents({}), "records")
```

### 4. Run training pipeline

```bash
python demo.py
```

### 5. Start Flask app

```bash
python app.py
# → http://localhost:5000
```

---

## Docker

```bash
# Build
docker build -t water-quality-predictor .

# Run
docker run -p 5000:5000 \
  -e MONGODB_URL="mongodb+srv://..." \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  -e AWS_DEFAULT_REGION="us-east-1" \
  water-quality-predictor
```

---

## AWS CI/CD Deployment

### Step 1 — Create IAM user with policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`
- `AmazonS3FullAccess`

### Step 2 — Create ECR repository

```bash
aws ecr create-repository --repository-name water-quality-repo --region us-east-1
# Save URI: <account-id>.dkr.ecr.us-east-1.amazonaws.com/water-quality-repo
```

### Step 3 — Create S3 bucket for model registry

```bash
aws s3 mb s3://water-quality-model-bucket
```

### Step 4 — Launch EC2 (Ubuntu 22.04, t2.medium+)

Install Docker on EC2:
```bash
sudo apt-get update -y && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 5 — Register EC2 as GitHub self-hosted runner

```
GitHub repo → Settings → Actions → Runners → New self-hosted runner
Select Linux → run commands shown
```

### Step 6 — Add GitHub Secrets

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPOSITORY_NAME` | `water-quality-repo` |
| `MONGODB_URL` | MongoDB Atlas connection string |

### Step 7 — Push to main → auto-deploys!

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Prediction UI |
| `/predict` | POST | Returns potability prediction |
| `/train` | GET | Triggers full training pipeline |

---

## Model Performance

| Metric | Score |
|---|---|
| F1 Score | ~0.74 |
| ROC-AUC | ~0.79 |
| Precision | ~0.73 |
| Recall | ~0.75 |

*Scores vary with SMOTETomek resampling. Imbalanced dataset (39% potable).*

---

## License

MIT
