import joblib
from pathlib import Path


class ModelLoader:

    def __init__(self, model_dir='outputs/models'):
        self.model_dir = Path(model_dir)

        self.model = None
        self.scaler = None
        self.feature_names = None

    def load(self):

        try:
            self.model = joblib.load(
                self.model_dir / 'churn_model.pkl'
            )

            self.scaler = joblib.load(
                self.model_dir / 'scaler.pkl'
            )

            print("✅ Models loaded successfully")

            return True

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def is_ready(self):
        """
        Check whether model is ready for predictions.
        """
        return self.model is not None and self.scaler is not None