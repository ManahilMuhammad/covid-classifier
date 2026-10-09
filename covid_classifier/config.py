from pathlib import Path

# dataset
KAGGLE_DATASET = "tawsifurrahman/covid19-radiography-database"

# classes
CLASSES = ["normal", "viral", "covid"]
CLASS_DIRS = {"normal": "Normal", "viral": "Viral Pneumonia", "covid": "COVID"}
CLASS_LABELS = {"normal": "Normal", "viral": "Viral pneumonia", "covid": "COVID-19"}

# image specifications
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV = ROOT / "data" / "splits.csv"
CHECKPOINT = ROOT / "checkpoints" / "best.pt"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
