import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def exponential_scc(y_true, y_pred):
    # Standard sparse categorical crossentropy for the base loss
    base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Calculate the numeric deviation between true and predicted class labels
    deviation = tf.abs(tf.argmax(y_true, axis=1) - tf.argmax(y_pred, axis=1))
    deviation = tf.cast(deviation, tf.float32)

    # Apply your specific penalty to the deviation, extreme penalty
    deviation_penalty = deviation * 1

    # Since base_loss is a tensor, we ensure deviation_penalty has the same shape for addition
    deviation_penalty = tf.reduce_mean(
        deviation_penalty, axis=-1
    )  # Adjust axis if necessary

    # Combine the base loss and the penalty
    return base_loss + deviation_penalty


def reverse_category(category):
    if category == 28:
        return 0.7, 1.0

    start = category * 0.025
    end = start + 0.025

    return start, end


class CategoryRangeFinder:
    def __init__(self, predictions):
        self.predictions = predictions

    def find_smallest_range(self, threshold=0.8):
        best_range = None
        best_prob = None

        for category_start, prob in enumerate(self.predictions):
            current_prob = 0

            for category_end, prob in enumerate(self.predictions):
                if category_end <= category_start:
                    continue

                current_prob += prob

                if best_range is not None:
                    if current_prob > threshold:
                        if category_end - category_start < best_range[1] - best_range[
                            0
                        ] or (
                            category_end - category_start
                            == best_range[1] - best_range[0]
                            and current_prob > best_prob
                        ):
                            best_range = category_start, category_end
                            best_prob = current_prob

                        break

                    if category_end - category_start > best_range[1] - best_range[0]:
                        break
                else:
                    if current_prob > threshold:
                        best_range = category_start, category_end
                        best_prob = current_prob

        if best_prob is None or best_range is None:
            return (0.0, 1.0), 1.0

        rv_range = (
            reverse_category(best_range[0])[0],
            reverse_category(best_range[1])[1],
        )

        return rv_range, best_prob


class ReservationValuePredictor(object):
    model = None

    def __init__(self):
        # Recreate the model architecture
        self.model = load_model(
            os.path.dirname(__file__) + "/model/model1-v4_phase2.keras",
            custom_objects={"exponential_scc": exponential_scc},
        )

        # Recreate scaler
        self.history_scaler = joblib.load(
            os.path.dirname(__file__) + "/model/model1-v4_history_scaler.joblib"
        )
        self.features_scaler = joblib.load(
            os.path.dirname(__file__) + "/model/model1-v4_features_scaler.joblib"
        )

    def predict(
        self,
        history,
        avg,
        std_dev,
        minimum,
        maximum,
        change_rate,
        trend_direction,
        concession_end_estimate,
        progression,
        opp_nash_point_util,
        certainty_minimum=0.8,
    ):
        # Assuming the transformation functions and padding return correctly shaped arrays
        history_set = np.array([history])
        history_set = self.transform_history(history_set)
        history_set = pad_sequences(
            history_set, maxlen=100, dtype="float32", padding="post", truncating="post"
        )

        features = [
            avg,
            std_dev,
            minimum,
            maximum,
            change_rate,
            trend_direction,
            concession_end_estimate,
            progression,
            opp_nash_point_util,
        ]

        features_set = np.array([features])
        features_set = self.transform_features(features_set)

        # Make sure the inputs are numpy arrays of the correct shape
        prediction = self.model.predict(
            [np.array(history_set), np.array(features_set)], verbose=0
        )[0]

        range_finder = CategoryRangeFinder(prediction)

        prediction_value = (
            reverse_category(np.argmax(prediction))[1],
            prediction[np.argmax(prediction)],
        )
        prediction_range = range_finder.find_smallest_range(certainty_minimum)

        return prediction_value, prediction_range

    def transform_history(self, history_set):
        history_set_flat = history_set.reshape(-1, history_set.shape[-1])
        history_set_flat = self.history_scaler.fit_transform(history_set_flat)

        return history_set_flat.reshape(history_set.shape)

    def transform_features(self, features_set):
        return self.features_scaler.transform(features_set)
