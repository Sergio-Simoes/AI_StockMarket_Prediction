from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
import numpy as np


class Accuracy:
    def mean_forecast_error(self, expected_values, predicted_values):
        forecast_error = [expected_values[i] - predicted_values[i] for i in range(len(expected_values))]
        return sum(forecast_error) * 1.0 / len(expected_values)

    def mae(self, expected_values, predicted_values):
        return mean_absolute_error(expected_values, predicted_values)

    def mse(self, expected_values, predicted_values):
        return mean_squared_error(expected_values, predicted_values)

    def rmse(self, mse):
        return sqrt(mse)

    def mape(self, expected_values, predicted_values):
        expected_array = np.array(expected_values)
        predicted_array = np.array(predicted_values)
        return np.mean(np.abs((expected_array - predicted_array) / expected_array)) * 100

    def print_all(self, expected, predicted, convert):
        predicted_values = [item for sublist in predicted for item in sublist] if convert else predicted
        expected_values = [item for sublist in expected for item in sublist][:len(predicted_values)] if convert else expected[:len(predicted_values)]

        mean_error = self.mean_forecast_error(expected_values, predicted_values)
        mae = self.mae(expected_values, predicted_values)
        mse = self.mse(expected_values, predicted_values)
        rmse = self.rmse(mse)
        mape = self.mape(expected_values, predicted_values)

        print("Mean Forecast Error: ", mean_error)
        print("Mean Absolute Error: ", mae)
        print("Mean Squared Error: ", mse)
        print("Root Mean Squared Error: ", rmse)
        print("Mean Absolute Percentage Error: ", mape)

        return expected_values, predicted_values
