import os
import random
import numpy as np
from PIL import Image
import cv2
import json
import time

DIR_TASK = r"./tasks/task-z"
FILENAME_TASK_JSON = r"task.json"
FILEPATH_INPUT_DATA_JSON = r"train/dataset.json"


def answer_data_type(answer_value_type):
    if answer_value_type == "integer":
        return int
    elif answer_value_type == "real":
        return float
    else:
        return None


def read_task(path_json):
    json_open = open(path_json, 'r', encoding='utf-8')
    task = json.load(json_open)

    return task["info"]


def read_dataset(path_json, answer_value_type=int, multi_data=False, input_data_type="image-3ch"):
    json_open = open(path_json, 'r')
    dataset = json.load(json_open)

    filename_list = []
    input_data_list = [] #入力データ
    correct_list = [] #正解値
    num_problem = 0

    for item in dataset["data"]:
        try:
            # 正解値
            correct_list.append(answer_value_type(item["gt"]))

            # 入力データ
            data = []
            filename = []
            if multi_data:
                for path in item["path"]:
                    # 画像読み込み
                    filename.append(path)
                    data.append(np.array(Image.open(os.path.join(os.path.dirname(path_json), path))))
            else:
                # 画像読み込み
                filename = item["path"]
                data.append(np.array(Image.open(os.path.join(os.path.dirname(path_json), item["path"]))))

            input_data = np.array(data, dtype=data[0].dtype)
            input_data_list.append(input_data)
            filename_list.append(filename)

            num_problem += 1
        except:
            print(f"入力データ({num_problem})の読み込みに失敗しました。")
            continue

    # 正解値リストをnumpyに変換
    correct_list = np.array(correct_list, dtype=answer_value_type)

    return num_problem, filename_list, input_data_list, correct_list


def recognition(input_data) -> float:
    # ここに処理を書く
    time.sleep(0.9 * input_data.shape[0])

    # floatで推定値を返す
    return np.random.uniform(0, 2)


def main():
    # タスク情報読み込み
    task = read_task(os.path.join(DIR_TASK, FILENAME_TASK_JSON))
    answer_value_type = answer_data_type(task["answer_value_type"])
    if answer_value_type is None:
        return

    # データ読み込み
    num_problem, filename_list, input_data_list, correct_list = read_dataset(
        os.path.join(DIR_TASK, FILEPATH_INPUT_DATA_JSON), answer_value_type, task["multi_input_data"], task["input_data_type"])

    # ユーザ作成の処理にかける
    answer_list = np.zeros((num_problem), answer_value_type)
    for i in range(num_problem):
        answer_list[i] = recognition(input_data_list[i])

    # 評価
    print("correct_list", correct_list)
    print("answer_list", answer_list)

    if task["metric"] == "MAE":
        evaluate = np.abs(answer_list - correct_list)
        print("evaluate", evaluate)
        print(f"MAE: {np.average(evaluate)}")
    elif task["metric"] == "Accuracy":
        evaluate = answer_list == correct_list
        print("evaluate", evaluate)
        print(f"Accuracy: {np.sum(evaluate) / num_problem}")
    

if __name__ == "__main__":
    main()
