import json
import os
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()


class Tester:
    def __init__(self, idToken, logId):
        self.idToken = idToken
        self.logId = logId
        self.timestamp = []
        self.price = []
        self.__parse_input_string__(self.__getPNL__(), self.timestamp, self.price)

    def plot_timestamp_value(self):
        # Plot the graph
        plt.plot(self.timestamp, self.price)
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.show()

    def __getPNL__(self):
        header = {
            'Authorization': f'Bearer {self.idToken}'
        }

        response = requests.get(
            f"https://bz97lt8b1e.execute-api.eu-west-1.amazonaws.com/prod/results/tutorial/{self.logId}",
            headers=header)

        response_json = json.loads(response.text)
        return response_json["algo"]["summary"]["graphLog"][
               response_json["algo"]["summary"]["graphLog"].find("\n") + 1:]

    @staticmethod
    def __parse_input_string__(input_string, timestamp_list, price_list):
        split_input = input_string.strip().split('\n')
        for i in range(0, len(split_input), 2):
            parts = split_input[i].split(';')
            timestamp_list.append(parts[0])
            price_list.append(float(parts[1]))
        return timestamp_list, price_list


test = Tester(os.getenv('PROSPERITY_ID_TOKEN'), os.getenv('LOG_ID'))
test.plot_timestamp_value()
