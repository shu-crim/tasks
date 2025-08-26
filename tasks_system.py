import os
import gc
import numpy as np
import importlib
from PIL import Image
from matplotlib import pyplot as plt
import glob
import shutil
import datetime
import time
from enum import Enum
from multiprocessing import Pool, TimeoutError
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import chardet
import random
import cv2
import sklearn.metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix  
from collections import defaultdict
from module.task import Task, Log
from module.image import ImageProc


FILENAME_DATASET_JSON = r"dataset.json"
PROC_TIMEOUT_SEC = 1


def UpdateTtimestamp(task_id):
    # ディレクトリが無ければ作成
    if not os.path.exists(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME)):
        os.makedirs(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME))

    with open(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, Task.TIMESTAMP_FILE_NAME), "w", encoding='utf-8') as f:
        f.write(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))


def read_dataset(path_json, answer_value_type=int, multi_data:bool=False, input_data_type:Task.InputDataType=Task.InputDataType.Image3ch):
    json_open = open(path_json, 'r', encoding='utf-8')
    dataset = json.load(json_open)

    filename_list = []
    input_data_list = [] #入力データ
    parameter_list = [] #パラメータ
    correct_list = [] #正解値
    num_problem = 0
    parameter_type = []

    # パラメータ型の読み込み
    if "parameter_type" in dataset:
        types = dataset["parameter_type"]
        for t in types:
            if t == "real":
                parameter_type.append(Task.ParameterType.real)
            elif t == "integer":
                parameter_type.append(Task.ParameterType.integer)

    # データの読み込み
    for item in dataset["data"]:
        try:
            # 正解値
            if answer_value_type == Task.AnswerValueType.Image1ch:
                # 画像の読み込み(グレースケール)
                correct_list.append(np.array(Image.open(os.path.join(os.path.dirname(path_json), item["gt"])).convert("L")))
            elif answer_value_type == Task.AnswerValueType.Image3ch:
                # 画像の読み込み(カラー)
                correct_list.append(np.array(Image.open(os.path.join(os.path.dirname(path_json), item["gt"]))))
            elif answer_value_type == Task.AnswerValueType.real:
                correct_list.append(float(item["gt"]))
            elif answer_value_type == Task.AnswerValueType.integer:
                correct_list.append(int(item["gt"]))
            elif answer_value_type == Task.AnswerValueType.ActiveLearing:
                correct_list.append(item["gt"])
            elif answer_value_type == Task.AnswerValueType.FeatureExtraction:
                correct_list.append(item["gt"])

            # 入力データ
            data = []
            filename = []
            if input_data_type == Task.InputDataType.Vector:
                # ベクトル
                data = np.array(item["vector"], np.float32)
                if "path" in item:
                    filename = item["path"]
                else:
                    filename = ["" for i in range(len(data))]
            else:
                # 画像
                if multi_data:
                    for path in item["path"]:
                        # 画像読み込み
                        filename.append(path)
                        image = Image.open(os.path.join(os.path.dirname(path_json), path))
                        if input_data_type == Task.InputDataType.Image1ch:
                            image = image.convert("L")
                        image = np.array(image)
                        if answer_value_type == Task.AnswerValueType.FeatureExtraction:
                            if input_data_type == Task.InputDataType.Image3ch:
                                # 入力画像が2次元（グレースケール）の場合、3次元に拡張する(H, W) -> (H, W, 3)
                                if image.ndim == 2:
                                    image = np.stack([image] * 3, axis=-1)
                        data.append(image)
                else:
                    # 画像読み込み
                    filename = item["path"]
                    image = Image.open(os.path.join(os.path.dirname(path_json), item["path"]))
                    if input_data_type == Task.InputDataType.Image1ch:
                        image = image.convert("L")
                    data = np.array(image)

            # 複数データの場合にひとまとめの行列にする
            if multi_data:
                if answer_value_type == Task.AnswerValueType.FeatureExtraction: # Feature Extraction Taskの場合はそのまま
                    input_data_list.append(data)
                else:
                    input_data_list.append(np.array(data, dtype=data[0].dtype))
            else:
                input_data_list.append(data)

            if answer_value_type == Task.AnswerValueType.ActiveLearing:
                # Active learing Taskの場合、学習データをパラメータとして読み込む
                parameter = [[], [], 0] #[入力データのリスト, ラベルのリスト, P/R目標値]
                if input_data_type == Task.InputDataType.Vector:
                    # ベクトル読み込み
                    parameter[0] = np.array(item["train"], np.float32)
                else:
                    raise(ValueError(f"answer_value_typeの指定({answer_value_type})が不正です。"))
                
                # ラベルと目標値の読み込み
                parameter[1] += item["train-gt"]
                parameter[2] = item["goal"]

                # データ追加
                parameter_list.append(parameter)
            elif answer_value_type == Task.AnswerValueType.FeatureExtraction:
                # Feature Extraction Taskの場合、shot数をパラメータとして読み込む
                parameter = [item["shots"], item["try"]] #[shot数, 試行数]
                parameter_list.append(parameter)
            else:
                # 通常のパラメータ読み込み
                param_list = []
                if "parameter" in item:
                    params = item["parameter"]
                    for index in range(len(parameter_type)):
                        if parameter_type[index] == Task.ParameterType.real:
                            param_list.append(float(params[index]))
                        if parameter_type[index] == Task.ParameterType.integer:
                            param_list.append(int(params[index]))
                parameter_list.append(param_list)

            # ファイル名を追加
            filename_list.append(filename)

            num_problem += 1
        except Exception as e:
            print(f"入力データ({num_problem})の読み込みに失敗しました： {e}")
            continue

    return num_problem, filename_list, input_data_list, parameter_list, correct_list


def evaluate(num_problem, input_data_list, parameter_list, func_recognition, answer_value_type:Task.AnswerValueType, timelimit_per_data=PROC_TIMEOUT_SEC, attachment_path:str=None):
    total_proc_time = 0
    try:
        # ユーザ作成の処理にかける
        answer_list = []
        with Pool(processes=1) as p:
            for i in range(num_problem):
                if answer_value_type == Task.AnswerValueType.FeatureExtraction:
                    num_input_data = len(input_data_list[i]) # 特徴抽出タスクの場合、input_data_list[i]は画像のリスト
                else:
                    num_input_data = input_data_list[i].shape[0]
                time_limit = timelimit_per_data * (num_input_data + 20) if i == 0 else timelimit_per_data * num_input_data # 初回のみオーバーヘッドを考慮してゆるめ

                start_time = time.time()

                if answer_value_type == Task.AnswerValueType.FeatureExtraction:
                    # 特徴抽出タスクの場合、parameter_list[i]は[shot数, 試行数]のリスト。ユーザ処理へは添付ファイルへのパスを渡す
                    apply_result = p.apply_async(func_recognition, (input_data_list[i], attachment_path))      
                elif len(parameter_list[i]) == 0:
                    apply_result = p.apply_async(func_recognition, (input_data_list[i],))
                elif len(parameter_list[i]) == 1:
                    apply_result = p.apply_async(func_recognition, (input_data_list[i], parameter_list[i][0],))
                elif len(parameter_list[i]) == 2:
                    apply_result = p.apply_async(func_recognition, (input_data_list[i], parameter_list[i][0], parameter_list[i][1],))
                elif len(parameter_list[i]) == 3:
                    apply_result = p.apply_async(func_recognition, (input_data_list[i], parameter_list[i][0], parameter_list[i][1], parameter_list[i][2],))
                else:
                    apply_result = p.apply_async(func_recognition, (input_data_list[i], parameter_list[i],))

                # apply_result = p.apply_async(func_recognition, (input_data_list[i],))
                answer = apply_result.get(timeout=time_limit)

                # 返り値の型を矯正
                # answer = answer_value_type(answer)
                if answer_value_type == Task.AnswerValueType.integer:
                    answer = int(answer)
                elif answer_value_type == Task.AnswerValueType.real:
                    answer = float(answer)
                elif answer_value_type == Task.AnswerValueType.Image1ch:
                    answer = np.array(answer, dtype=np.uint8)
                elif answer_value_type == Task.AnswerValueType.Image3ch:
                    answer = np.array(answer, dtype=np.uint8)
                elif answer_value_type == Task.AnswerValueType.ActiveLearing:
                    answer = np.array(answer, dtype=int)
                elif answer_value_type == Task.AnswerValueType.FeatureExtraction:
                    answer = np.array(answer, dtype=float)
                
                end_time = time.time()
                total_proc_time += end_time - start_time
                #print(f'proc time: {end_time - start_time} s')
                if end_time - start_time > time_limit:
                    raise(TimeoutError("処理がタイムアウトしました。"))

                answer_list.append(answer)

    except TimeoutError:
        print("処理がタイムアウトしました。")
        raise(TimeoutError("処理がタイムアウトしました。"))
    except Exception as e:
        raise(e)
    
    return answer_list, total_proc_time


class Result:
    data_type = Task.DataType.train
    filename = ""
    correct = 0
    answer = 0
    inputdata = None
    parameter = None

    def __init__(self, data_type, filename, correct, answer, inputdata, parameter) -> None:
        self.data_type = data_type
        self.filename = filename
        self.correct = correct
        self.answer = answer
        self.inputdata = inputdata
        self.parameter = parameter
    

def evaluate3data(task_id, module_name, user_name, answer_value_type:Task.AnswerValueType, multi_data:bool=False, data_type:Task.InputDataType=Task.InputDataType.Image3ch, contest:bool=False, timelimit_per_data=PROC_TIMEOUT_SEC, attachment_path:str=None):
    try:
        # ユーザ作成の処理を読み込む
        user_module = importlib.import_module(f"{Task.TASKS_DIR}.{task_id}.{Task.USER_MODULE_DIR_NAME}.{module_name}")
    except:
        raise(ValueError("モジュールを読み込めません。"))
    
    try:
        # ユーザ作成の処理を読み込む
        func_recognition = user_module.recognition
    except:
        raise(ValueError("関数recognition()を読み込めません。"))
    
    # 結果のリスト
    result_list = []
    
    start = time.time()

    try:
        # train
        num_train, filename_list, input_data_list, parameter_list, correct_list = read_dataset(
            os.path.join(Task.TASKS_DIR, task_id, "train", FILENAME_DATASET_JSON), answer_value_type, multi_data, data_type)

        answer_list, total_proc_time = evaluate(num_train, input_data_list, parameter_list, func_recognition, answer_value_type, timelimit_per_data, attachment_path)
        if num_train == 0:
            return []
        for i in range(num_train):
            # 画像の場合はサイズのチェック
            if answer_value_type == Task.AnswerValueType.Image1ch or answer_value_type == Task.AnswerValueType.Image3ch:
                if correct_list[i].shape != answer_list[i].shape:
                    raise(ValueError("処理結果の値が適切ではありません。"))

            result = Result(Task.DataType.train, filename_list[i], correct_list[i], answer_list[i], input_data_list[i], parameter_list[i])
            result_list.append(result)
        print(f'Train({user_name}) average proc time: {total_proc_time / num_train : .1f}s, total: {total_proc_time : .1f} s')

        # メモリ解放
        if answer_value_type == Task.AnswerValueType.FeatureExtraction:
            del input_data_list
            gc.collect()

        # valid
        num_valid, filename_list, input_data_list, parameter_list, correct_list = read_dataset(
            os.path.join(Task.TASKS_DIR, task_id, "valid", FILENAME_DATASET_JSON), answer_value_type, multi_data, data_type)
        
        # シャッフル
        rng = np.random.default_rng(int(start))
        rng.shuffle(input_data_list)
        rng = np.random.default_rng(int(start))
        rng.shuffle(parameter_list)
        rng = np.random.default_rng(int(start))
        rng.shuffle(correct_list)
        rng = np.random.default_rng(int(start))
        rng.shuffle(filename_list)

        answer_list, total_proc_time = evaluate(num_valid, input_data_list, parameter_list, func_recognition, answer_value_type, timelimit_per_data, attachment_path)
        if num_valid == 0:
            return []
        for i in range(num_valid):
            # 画像の場合はサイズのチェック
            if answer_value_type == Task.AnswerValueType.Image1ch or answer_value_type == Task.AnswerValueType.Image3ch:
                if correct_list[i].shape != answer_list[i].shape:
                    raise(ValueError("処理結果の値が適切ではありません。"))

            result = Result(Task.DataType.valid, filename_list[i], correct_list[i], answer_list[i], input_data_list[i], parameter_list[i])
            result_list.append(result)
        print(f'Valid({user_name}) average proc time: {total_proc_time / num_valid : .1f}s, total: {total_proc_time : .1f} s')

        # メモリ解放
        if answer_value_type == Task.AnswerValueType.FeatureExtraction:
            del input_data_list
            gc.collect()

        # test
        if contest:
            num_test, filename_list, input_data_list, parameter_list, correct_list = read_dataset(
                os.path.join(Task.TASKS_DIR, task_id, "test", FILENAME_DATASET_JSON), answer_value_type, multi_data, data_type)
        
            # シャッフル
            rng = np.random.default_rng(int(start))
            rng.shuffle(input_data_list)
            rng = np.random.default_rng(int(start))
            rng.shuffle(parameter_list)
            rng = np.random.default_rng(int(start))
            rng.shuffle(correct_list)
            rng = np.random.default_rng(int(start))
            rng.shuffle(filename_list)

            answer_list, total_proc_time = evaluate(num_test, input_data_list, parameter_list, func_recognition, answer_value_type, timelimit_per_data, attachment_path)
            if num_test == 0:
                return []
            for i in range(num_test):
                # 画像の場合はサイズのチェック
                if answer_value_type == Task.AnswerValueType.Image1ch or answer_value_type == Task.AnswerValueType.Image3ch:
                    if correct_list[i].shape != answer_list[i].shape:
                        raise(ValueError("処理結果の値が適切ではありません。"))

                result = Result(Task.DataType.test, filename_list[i], correct_list[i], answer_list[i], input_data_list[i], parameter_list[i])
                result_list.append(result)
            print(f'Test({user_name}) average proc time: {total_proc_time / num_test : .1f}s, total: {total_proc_time : .1f} s')

            # メモリ解放   
            if answer_value_type == Task.AnswerValueType.FeatureExtraction:
                del input_data_list
                gc.collect()

    except Exception as e:
        raise(e)

    print(f'Proc Time({user_name}): {time.time()-start : .1f} s')

    return result_list


def caclActiveLearingAccuracy(label_gt, label_estimation, num_class):
    precision = sklearn.metrics.precision_score(label_gt, label_estimation, average=None, labels=[i for i in range(num_class)], zero_division=1)
    recall = sklearn.metrics.recall_score(label_gt, label_estimation, average=None, labels=[i for i in range(num_class)], zero_division=1)

    return precision, recall


def evaluateActiveLearing(num_class:int, train_data:list, label_train:list, valid_data:list, label_valid:list, registration_order:list, goal:float) -> float | list:
    num_registration_per_valid = 5 #5データずつ登録して評価
    num_valid_data = len(registration_order)
    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    label_train = np.array(label_train)
    label_valid = np.array(label_valid)
    K = 1
    rr_detail = [] #0%～最大100%登録時点での各クラスのP/R
    registration_rate = -1

    # 評価と登録のループ
    for iValidObject in range(len(registration_order) + 1):
        if iValidObject % num_registration_per_valid == 0 or iValidObject == num_valid_data:
            if len(label_train) > 0:
                # Trainデータの登録
                knn = cv2.ml.KNearest_create()
                knn.train(train_data, cv2.ml.ROW_SAMPLE, label_train)

                # Validデータの識別
                ret, results, neighbours, dist = knn.findNearest(valid_data, K)
                label_estimation = results.reshape(label_valid.shape[0]).astype(int)
            else:
                # Validデータの識別結果はすべてclass0
                label_estimation = np.zeros((label_valid.shape[0]), int)

            precision, recall = caclActiveLearingAccuracy(label_valid, label_estimation, num_class)

            # 詳細記録
            if iValidObject == 0:
                rr_detail.append([precision, recall])
            else:
                registration_rate_percent = int((iValidObject / num_valid_data) * 100)
                while len(rr_detail) < registration_rate_percent + 1:
                    rr_detail.append(rr_detail[-1]) #最後に登録された結果をコピーする
                rr_detail[registration_rate_percent] = [precision, recall]

            # Goal達成判定
            if min(precision) >= goal and min(recall) >= goal and registration_rate < 0:
                print(f"P/R Goal have been achieved! {iValidObject} / {num_valid_data}")
                registration_rate = iValidObject / num_valid_data

        # trainデータの追加
        if iValidObject < num_valid_data:
            # リストから順に選択
            addition_index = registration_order[iValidObject]

            # Trainデータに追加
            additional_data = valid_data[addition_index]
            train_data = np.concatenate([train_data, [additional_data]])
            label_train = np.concatenate([label_train, [label_valid[addition_index]]])
            # print(f"{iValidObject}:valid data[{addition_index}] was added.")

    return registration_rate, rr_detail if len(rr_detail) > 0 else None


def evaluateFeatureExtraction(features: np.ndarray, corrects: np.ndarray, num_shots: int, num_try: int) -> tuple[float, dict]:
    """
    特徴量の識別性能をN-shotの最近傍法で評価し、マクロ平均F1スコアと詳細な評価結果を返す。
    num_try回試行し、その平均スコアを算出する。

    Args:
        features (np.ndarray): 特徴量集合 (画像数 x 特徴次元数)。
        corrects (np.ndarray): 各特徴量に対応する正解クラスラベル (画像数,)。
        num_shots (int): 各クラスから登録データとしてランダムに選択するサンプル数。
        num_try (int): 評価の試行回数。シードを0からnum_try-1まで変えて実行する。

    Returns:
        tuple[float, dict]:
            - float: num_try回試行したマクロ平均F1スコアの算術平均。
            - dict: 各クラスのprecision, recall, f1-score, gt, tp, fn, fpの平均値を格納した辞書。
                    キーはクラスラベル、値は評価指標の辞書。
                    例: {
                        0: {'precision': 0.8, 'recall': 0.9, 'f1': 0.85, 'gt': 50, 'tp': 45, 'fn': 5, 'fp': 11},
                        ...
                    }
    """
    
    # 各試行のマクロF1スコアを格納するリスト
    f1_scores = []
    # 各クラスの詳細な評価指標の合計値を格納する辞書 (gt, tp, fn, fp を追加)
    detail_sum = defaultdict(lambda: {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'gt': 0.0, 'tp': 0.0, 'fn': 0.0, 'fp': 0.0
    })
    unique_classes = np.unique(corrects)

    # num_tryの回数だけ評価を繰り返す
    for i in range(num_try):
        np.random.seed(i)

        # Step 1: 登録データ(gallery)を作成する
        # ----------------------------------------------------------------------
        gallery_features_list = []
        gallery_labels_list = []

        for class_id in unique_classes:
            class_indices = np.where(corrects == class_id)[0]
            # クラスのサンプル数がnum_shotsより少ない場合も考慮
            n_to_select = min(num_shots, len(class_indices))
            if n_to_select == 0:
                continue
            
            # np.random.seed(i) の影響を受け、この選択が試行ごとに変わる
            gallery_indices = np.random.choice(class_indices, size=n_to_select, replace=False)
            
            gallery_features_list.append(features[gallery_indices])
            gallery_labels_list.extend([class_id] * n_to_select)

        # この試行で登録データが一つも作成できなかった場合は、スコアを0として次へ
        if not gallery_features_list:
            f1_scores.append(0.0)
            # 詳細スコアは何も加算しない
            continue

        gallery_features = np.vstack(gallery_features_list)
        gallery_labels = np.array(gallery_labels_list)

        # Step 2: 識別処理 (最近傍法 + コサイン類似度)
        # ----------------------------------------------------------------------
        similarities = cosine_similarity(features, gallery_features)
        nearest_indices = np.argmax(similarities, axis=1)
        predicted_labels = gallery_labels[nearest_indices]

        # Step 3: 評価
        # ----------------------------------------------------------------------
        # 3-1. マクロ平均F1スコアの算出
        trial_f1_score = f1_score(y_true=corrects, y_pred=predicted_labels, average='macro', zero_division=0)
        f1_scores.append(trial_f1_score)
        
        # 3-2. 各クラスの詳細な評価指標の算出と加算
        # support(s)も受け取るように変更
        p, r, f, s = precision_recall_fscore_support(
            y_true=corrects, 
            y_pred=predicted_labels, 
            labels=unique_classes, 
            average=None, # クラスごとに算出
            zero_division=0
        )
        
        # 混同行列を計算
        cm = confusion_matrix(
            y_true=corrects,
            y_pred=predicted_labels,
            labels=unique_classes
        )
        
        for idx, class_id in enumerate(unique_classes):
            detail_sum[class_id]['precision'] += p[idx]
            detail_sum[class_id]['recall'] += r[idx]
            detail_sum[class_id]['f1'] += f[idx]
            detail_sum[class_id]['gt'] += s[idx]  # support が GT数
            
            # 混同行列からTP, FN, FPを計算
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            
            detail_sum[class_id]['tp'] += tp
            detail_sum[class_id]['fn'] += fn
            detail_sum[class_id]['fp'] += fp
    
    # 実際に評価が行われた試行回数を取得
    valid_tries = len(f1_scores)

    # 試行が一度も有効に実行されなかった場合
    if valid_tries == 0:
        # GT数は事前に計算できるが、他の指標と合わせるため0で初期化
        detail = {
            class_id: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'gt': 0.0, 'tp': 0.0, 'fn': 0.0, 'fp': 0.0}
            for class_id in unique_classes
        }
        return 0.0, detail
    
    # マクロF1スコアの平均値を計算
    average_f1 = np.mean(f1_scores)
    
    # 詳細評価指標の平均値を計算
    detail = {}
    for class_id in sorted(detail_sum.keys()):
        detail[class_id] = {
            'precision': detail_sum[class_id]['precision'] / valid_tries,
            'recall': detail_sum[class_id]['recall'] / valid_tries,
            'f1': detail_sum[class_id]['f1'] / valid_tries,
            'gt': detail_sum[class_id]['gt'] / valid_tries,
            'tp': detail_sum[class_id]['tp'] / valid_tries,
            'fn': detail_sum[class_id]['fn'] / valid_tries,
            'fp': detail_sum[class_id]['fp'] / valid_tries,
        }
    
    return average_f1, detail


def ProcOneUser(task_id, user_name, new_filename, attachment_path, now, memo=''):
    # タスク情報の読み込み
    task:Task = Task(task_id)
    if task is None:
        print("タスク情報を読み込めません: {task_id}")
        return

    # 処理と評価を実行
    proc_success = False
    message = ''
    try:
        result_list = evaluate3data(
            task_id, os.path.splitext(new_filename)[0], # 拡張子を除く
            user_name, task.answer_value_type, task.multi_input_data,
            task.input_data_type, True if task.type == Task.TaskType.Contest else False,
            task.timelimit_per_data, attachment_path)
        
        proc_success = True
    except Exception as e:
        proc_success = False
        message = e
        print(f'Error evaluate3data: {e}')

    if proc_success:
        # 評価結果を集計
        if task.metric == Task.Metric.Accuracy:
            num_true = {}
            num_false = {}
            for data_type in Task.DataType:
                num_true[data_type] = 0
                num_false[data_type] = 0

            for result in result_list:
                if result.correct == result.answer:
                    num_true[result.data_type] += 1
                else:
                    num_false[result.data_type] += 1
        elif task.metric == Task.Metric.MAE:
            abs_errors = {}
            for result in result_list:
                if not result.data_type in abs_errors:
                    abs_errors[result.data_type] = []

                if task.answer_value_type == Task.AnswerValueType.Image1ch or task.answer_value_type == Task.AnswerValueType.Image3ch:
                    # 画素ごとの絶対誤差を画像全体で平均
                    abs_error = np.average(np.abs(result.answer.astype(int) - result.correct.astype(int)))
                else:
                    # 値の絶対誤差
                    abs_error = np.abs(result.answer - result.correct)

                abs_errors[result.data_type].append(abs_error)
        elif task.metric == Task.Metric.RegistrationRate:
            registration_rate = {}
            rr_detail = {}
            for data_type in Task.DataType:
                registration_rate[data_type] = []
                rr_detail[data_type] = []

            for result in result_list:
                train_data, label_train, goal = result.parameter
                num_class = max(label_train) + 1
                rr, detail = evaluateActiveLearing(num_class, train_data, label_train, result.inputdata, result.correct, result.answer, goal)

                registration_rate[result.data_type].append(rr)
                if detail is not None:
                    rr_detail[result.data_type].append(detail)
        elif task.metric == Task.Metric.AverageF1Score:
            # 特徴抽出の評価
            average_f1_scores = {}
            average_f1_scores_detail = {}
            for result in result_list:
                num_shots, num_try = result.parameter
                average_f1_score, detail = evaluateFeatureExtraction(result.answer, result.correct, num_shots, num_try)
                if not result.data_type in average_f1_scores:
                    average_f1_scores[result.data_type] = []
                average_f1_scores[result.data_type].append(average_f1_score)
                if not result.data_type in average_f1_scores_detail:
                    average_f1_scores_detail[result.data_type] = []
                average_f1_scores_detail[result.data_type].append(detail)


        # 評価結果の詳細を出力
        output_csv_filename = user_name + "_" + now.strftime('%Y%m%d_%H%M%S') + ".csv"
        with open(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "detail", output_csv_filename), "w", encoding='utf-8') as output_csv_file:
            # 集計
            output_csv_file.write(f"filename,{os.path.basename(new_filename)}\n\n")
            if task.metric == Task.Metric.Accuracy:
                output_csv_file.write("type,num_data,true,false,accuracy\n")
            elif task.metric == Task.Metric.MAE:
                output_csv_file.write("type,num_data,MAE\n")   
            elif task.metric == Task.Metric.RegistrationRate:
                output_csv_file.write("type,num_data,RegistrationRate\n")
            elif task.metric == Task.Metric.AverageF1Score:
                output_csv_file.write("type,num_data,AverageF1Score\n")

            if task.metric == Task.Metric.AverageF1Score:
                for data_type in average_f1_scores.keys():
                    output_csv_file.write(f"{data_type.name},{len(average_f1_scores[data_type])},{np.average(np.array(average_f1_scores[data_type]))}\n")
            else:
                for data_type in Task.DataType:
                    if task.metric == Task.Metric.Accuracy:
                        output_csv_file.write(f"{data_type.name},{num_true[data_type] + num_false[data_type]},{num_true[data_type]},{num_false[data_type]},{num_true[data_type]/num_data if num_data > 0 else '-'}\n")
                    elif task.metric == Task.Metric.MAE:
                        if data_type in abs_errors:
                            output_csv_file.write(f"{data_type.name},{len(abs_errors[data_type])},{np.average(np.array(abs_errors[data_type], float))}\n")
                    elif task.metric == Task.Metric.RegistrationRate:
                        output_csv_file.write(f"{data_type.name},{len(registration_rate[data_type])},{np.average(np.array(registration_rate[data_type], float))}\n")

            # 詳細
            output_csv_file.write("\n")
            if task.metric == Task.Metric.Accuracy:
                output_csv_file.write("type,filename,correct,answer,check\n")
                for result in result_list:
                    output_csv_file.write(f"{result.data_type.name},{result.filename.replace(',', '-')},{result.correct},{result.answer},{1 if result.correct == result.answer else 0}\n")
            elif task.metric == Task.Metric.MAE:
                output_csv_file.write("type,filename,correct,answer,abs_error\n")
                for result in result_list:
                    if task.answer_value_type == Task.AnswerValueType.Image1ch or task.answer_value_type == Task.AnswerValueType.Image3ch:
                        correct = "-"
                        answer = "-"
                        # 画素ごとの絶対誤差を画像全体で平均
                        abs_error = np.average(np.abs(result.answer.astype(int) - result.correct.astype(int)))
                    else:
                        correct = result.correct
                        answer = result.answer
                        # 値の絶対誤差
                        abs_error = np.abs(result.answer - result.correct)
                    output_csv_file.write(f"{result.data_type.name},{str(result.filename).replace(',', '-')},{correct},{answer},{abs_error}\n")
            elif task.metric == Task.Metric.RegistrationRate:
                num_data = {}
                for data_type in Task.DataType:
                    num_data[data_type] = 0
                # 各出題の成績
                output_csv_file.write("type,index,RegistrationRate\n")
                for result in result_list:
                    output_csv_file.write(f"{result.data_type.name},{num_data[result.data_type]},{registration_rate[result.data_type][num_data[result.data_type]]}\n")
                    num_data[result.data_type] += 1
                # Train/Valid/Testそれぞれの登録率vsP/R
                for data_type in Task.DataType:
                    if len(rr_detail[data_type]) > 0:
                        # 登録率[0..100]に対する平均P/Rを算出
                        details = np.array(rr_detail[data_type])
                        mean_pr = np.average(details, axis=0)

                        output_csv_file.write("\n")
                        output_csv_file.write(f"detail,{data_type.name}\n")
                        output_csv_file.write("RegistrationRate,")
                        for iClass in range(mean_pr.shape[2]):
                            output_csv_file.write(f"precision-{iClass},")
                        output_csv_file.write("\n")
                        for iRR in range(mean_pr.shape[0]):
                            output_csv_file.write(f"{iRR},")
                            for iClass in range(mean_pr.shape[2]):
                                output_csv_file.write(f"{mean_pr[iRR, 0, iClass]:.03},")
                            output_csv_file.write("\n")
                        for iClass in range(mean_pr.shape[2]):
                            output_csv_file.write(f"recall-{iClass},")
                        output_csv_file.write("\n")
                        for iRR in range(mean_pr.shape[0]):
                            output_csv_file.write(f"{iRR},")
                            for iClass in range(mean_pr.shape[2]):
                                output_csv_file.write(f"{mean_pr[iRR, 1, iClass]:.03},")
                            output_csv_file.write("\n")
            elif task.metric == Task.Metric.AverageF1Score:
                output_csv_file.write("type,index,AverageF1Score\n")
                for data_type in average_f1_scores.keys():
                    for index in range(len(average_f1_scores[data_type])):
                        output_csv_file.write(f"{result.data_type.name},{index},{average_f1_scores[data_type][index]}\n")

                # 各クラスの詳細スコア
                output_csv_file.write("\n")
                for data_type in average_f1_scores_detail.keys():
                    for index in range(len(average_f1_scores_detail[data_type])):
                        detail = average_f1_scores_detail[data_type][index]
                        output_csv_file.write(f"detail,{data_type.name},{index}\n")
                        output_csv_file.write("class,GT,TP,FN,FP,precision,recall,f1-score\n")
                        for class_id in sorted(detail.keys()):
                            metrics = detail[class_id]
                            output_csv_file.write(
                                f"{class_id},"
                                f"{int(round(metrics.get('gt', 0)))},"  # GTは四捨五入して整数に
                                f"{metrics.get('tp', 0):.1f},"          # TPは四捨五入して小数点以下1桁に
                                f"{metrics.get('fn', 0):.1f},"          # FNは四捨五入して小数点以下1桁に
                                f"{metrics.get('fp', 0):.1f},"          # FPは四捨五入して小数点以下1桁に
                                f"{metrics.get('precision', 0.0):.4f},"
                                f"{metrics.get('recall', 0.0):.4f},"
                                f"{metrics.get('f1', 0.0):.4f}\n"
                            )
                        output_csv_file.write("\n")

    # ユーザ毎の結果出力
    csv_path = os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "user", user_name + ".csv")
    if not os.path.exists(csv_path):
        # ファイルが無いのでヘッダを付ける
        with open(csv_path, "w", encoding='utf-8') as output_csv_file:
            output_csv_file.write("date,time,filename,")

            for data_type in Task.DataType:
                if task.metric == Task.Metric.Accuracy:
                    output_csv_file.write(f"{data_type.name}_true,{data_type.name}_false,{data_type.name}_accuracy,")
                elif task.metric == Task.Metric.MAE:
                    output_csv_file.write(f"{data_type.name}_MAE,")
                elif task.metric == Task.Metric.RegistrationRate:
                    output_csv_file.write(f"{data_type.name}_RegistrationRate,")
                elif task.metric == Task.Metric.AverageF1Score:
                    output_csv_file.write(f"{data_type.name}_AverageF1Score,")

            output_csv_file.write("message,memo")

    with open(csv_path, "a", encoding='utf-8') as output_csv_file:
        output_csv_file.write('\n') # 各結果の最初に改行を入れる。前の不正終了を引きずらないため
        output_csv_file.write(now.strftime('%Y/%m/%d,%H:%M:%S,'))
        output_csv_file.write(os.path.basename(new_filename) + ",")
        for data_type in Task.DataType:
            if task.metric == Task.Metric.Accuracy:
                if proc_success:
                    if data_type in num_true and data_type in num_false:
                        num_data = num_true[data_type] + num_false[data_type]
                        output_csv_file.write(f"{num_true[data_type]},{num_false[data_type]},{num_true[data_type]/num_data if num_data > 0 else '-'},")
                    else:
                        output_csv_file.write(f"-,-,-,")
                else:
                    output_csv_file.write(f"-,-,-,")
            elif task.metric == Task.Metric.MAE:
                if proc_success:
                    if data_type in abs_errors:
                        num_data = len(abs_errors[data_type])
                        if num_data > 0:
                            output_csv_file.write(f"{np.average(np.array(abs_errors[data_type], float))},")
                        else:
                            output_csv_file.write("-,")
                    else:
                        output_csv_file.write("-,")
                else:
                    output_csv_file.write("-,")
            elif task.metric == Task.Metric.RegistrationRate:
                if proc_success:
                    if data_type in registration_rate:
                        output_csv_file.write(f"{np.average(np.array(registration_rate[data_type], float))},")
                    else:
                        output_csv_file.write("-,")
                else:
                    output_csv_file.write("-,")
            elif task.metric == Task.Metric.AverageF1Score:
                if proc_success:
                    if data_type in average_f1_scores:
                        output_csv_file.write(f"{np.average(np.array(average_f1_scores[data_type], float))},")
                    else:
                        output_csv_file.write("-,")
                else:
                    output_csv_file.write("-,")

        output_csv_file.write(f"{message},{memo}")

    # 処理中であることを示すファイルを削除
    try:
        os.remove(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "user", f"{user_name}_inproc"))
    except:
        print(f"{user_name}_inproc を削除できませんでした。")

    # タイムスタンプ更新
    UpdateTtimestamp(task_id)

    # Log
    Log.write(f'{new_filename} on {task_id} {"was" if proc_success else "was not" } completed.')


def GetEncodingType(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return chardet.detect(rawdata)['encoding']


def move_script_atomic(
    source_py_path: str,
    destination_dir: str,
    user_name: str,
    task_id: str,
    now: datetime.datetime
) -> tuple[bool, str | dict]:
    """
    スクリプト(.py)と関連ファイル(.txt, .attachment)をアトミックに移動します。

    .pyと.txtはUTF-8に再エンコードされ、.attachmentはそのままコピーされます。
    全ての処理が成功した場合のみ元のファイルが削除されます。
    途中でエラーが発生した場合は、全ての変更をロールバックし、元の状態を維持します。

    Args:
        source_py_path (str): 移動元の.pyファイルのフルパス。
        destination_dir (str): 移動先のディレクトリ。
        user_name (str): 新しいファイル名に使用するユーザー名。
        task_id (str): 新しいファイル名に使用するタスクID。

    Returns:
        tuple[bool, str | dict]:
            成功した場合: (True, {'py': new_py_path, 'txt': new_txt_path, 'attachment': new_attachment_path})
            失敗した場合: (False, エラーメッセージ)
    """
    # 1. 全ての関連ファイルのパスを事前に定義
    original_py_path = source_py_path
    original_txt_path = source_py_path + '.txt'
    original_attachment_path = source_py_path + '.attachment'

    new_base_filename = f"{user_name}_{task_id}_{now.strftime('%Y%m%d_%H%M%S')}_{os.path.basename(source_py_path)}"
    new_py_path = os.path.join(destination_dir, new_base_filename)
    new_txt_path = os.path.join(destination_dir, new_base_filename + '.txt')
    new_attachment_path = os.path.join(destination_dir, new_base_filename + '.attachment')
    
    new_paths = {
        'py': new_py_path,
        'txt': new_txt_path,
        'attachment': new_attachment_path,
        "memo_content": ""
    }

    copied_files = []
    source_files_to_delete = []

    try:
        # --- フェーズ1: コピー処理 ---
        
        # .pyファイルを文字コード変換しつつコピー
        print(f"Copying .py file with re-encoding: {original_py_path}")
        encoding = GetEncodingType(original_py_path)
        with open(original_py_path, 'r', encoding=encoding) as f_in:
            content = f_in.read()
        with open(new_py_path, 'w', encoding='utf-8') as f_out:
            f_out.write(content)
        copied_files.append(new_py_path)
        source_files_to_delete.append(original_py_path)

        # .txtファイルがあれば、文字コード変換しつつコピー
        if os.path.exists(original_txt_path):
            print(f"Copying .txt file with re-encoding: {original_txt_path}")
            encoding = GetEncodingType(original_txt_path)
            with open(original_txt_path, 'r', encoding=encoding) as f_in:
                content = f_in.read()
                new_paths['memo_content'] = content
            with open(new_txt_path, 'w', encoding='utf-8') as f_out:
                f_out.write(content)
            copied_files.append(new_txt_path)
            source_files_to_delete.append(original_txt_path)
        else:
            new_paths['txt'] = '' # 存在しない場合はパスを空にする

        # 添付ファイルがあれば、そのままコピー
        if os.path.exists(original_attachment_path):
            print(f"Copying attachment file: {original_attachment_path}")
            shutil.copy2(original_attachment_path, new_attachment_path)
            copied_files.append(new_attachment_path)
            source_files_to_delete.append(original_attachment_path)
        else:
            new_paths['attachment'] = '' # 存在しない場合はパスを空にする

        # --- フェーズ2: 元ファイルの削除処理 ---
        print("All copies successful. Deleting original files...")
        for file_path in source_files_to_delete:
            os.remove(file_path)
        print("Process completed successfully.")
        
        return True, new_paths

    except Exception as e:
        print(f"An error occurred during the process: {e}")
        print("Rolling back changes...")
        
        # --- ロールバック処理 ---
        for file_path in copied_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Rolled back (removed): {file_path}")
            except Exception as rollback_e:
                print(f"FATAL: Failed to remove temporary file during rollback: {file_path}. Error: {rollback_e}")
        
        return False, str(e)


def main():
    with ProcessPoolExecutor(max_workers=4) as proccess:
        while True:
            # ディレクトリの一覧を作成して走査
            dir_list_tasks = glob.glob(os.path.join(Task.TASKS_DIR, '**/'))
            for dir_task in dir_list_tasks:
                # ディレクトリ名を取得→タスクIDとして使う
                task_id = os.path.basename(os.path.dirname(dir_task))

                # ディレクトリの一覧を作成して走査
                dir_list_users = glob.glob(os.path.join(dir_task, Task.UPLOAD_DIR_NAME, '**/'))
                for dir_user in dir_list_users:  
                    # ディレクトリ名を取得→ユーザ名として使う
                    user_name = os.path.basename(os.path.dirname(dir_user))

                    # ユーザのディレクトリ内の.pyファイルの一覧を作成して走査
                    py_files = glob.glob(os.path.join(dir_user, "*.py"))

                    if len(py_files) == 0:
                        continue
                    
                    path = py_files[0] # 最初に発見したファイルのみを対象とする

                    # モジュール移動先が無ければ生成(新規Taskの実行時)
                    dir_user_module = os.path.join(Task.TASKS_DIR, task_id, Task.USER_MODULE_DIR_NAME)
                    if not os.path.exists(dir_user_module):
                        os.makedirs(dir_user_module)

                    # ユーザのモジュールディレクトリに移動する
                    now = datetime.datetime.now()
                    success, result = move_script_atomic(
                        source_py_path=path,
                        destination_dir=dir_user_module,
                        user_name=user_name,
                        task_id=task_id,
                        now=now,
                    )

                    if not success:
                        print(f"ファイルの移動に失敗：{user_name} in task {task_id}: {result}")
                        continue

                    # 移動に成功したら評価
                    print(f"pcoccess start: {user_name}")
                    print(f"{path} -> {result['py']}")
                    new_filename = os.path.basename(result['py'])
                    attachment_path = result['attachment'] if 'attachment' in result else ''
                    memo = result['memo_content'] if 'memo_content' in result else ''

                    # 出力先ディレクトリが存在しない場合は生成(新規Taskの実行時)
                    dir_output_user = os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "user")
                    if not os.path.exists(dir_output_user):
                        os.makedirs(dir_output_user)

                    dir_output_detail = os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "detail")
                    if not os.path.exists(dir_output_detail):
                        os.makedirs(dir_output_detail)

                    # 評価中であることを示すファイルを生成
                    with open(os.path.join(dir_output_user, f"{user_name}_inproc"), "w", encoding='utf-8') as f:
                        pass

                    # タイムスタンプ更新
                    UpdateTtimestamp(task_id)

                    # Log
                    Log.write(f'{user_name} submit {new_filename} to {task_id}, file movement succeeded, then the proccess started.')

                    # ファイルの移動に成功したらプロセス生成して処理開始
                    # proccess.submit(ProcOneUser, task_id, user_name, new_filename, now, memo)
                    ProcOneUser(task_id, user_name, new_filename, attachment_path, now, memo) # デバッグのため同期実行

            # 少し待つ
            time.sleep(1.0)
            print(".")

if __name__ == "__main__":
    main()
