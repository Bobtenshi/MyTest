import requests


def SendLine():
    url = "https://notify-api.line.me/api/notify"
    token = "NXOF52tUd8CQ0qAWVXhh7k1RXeaeOQ8T8YwSLd8c9wB"
    headers = {"Authorization": "Bearer " + token}

    message = "Python-code(Relearning_MVAE in NTT-lab) has finished."
    payload = {"message": message}

    r = requests.post(url, headers=headers, params=payload)


if __name__ == "__main__":
    SendLine()
