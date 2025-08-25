import os
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix 
from collections import defaultdict
from typing import Tuple, Dict, Any


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from PIL import Image
import os

def recognition(input_data: list, attachment_path: str) -> np.ndarray:
    # 関数名、引数、戻り値の定義は変更しないでください。
    # input_data: numpyの入力画像(縦, 横, 3ch)のList
    # attachment_path: 添付アップロードされたファイルのパス

    # ResNetをベースとした特徴抽出ネットワークを定義
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            # ResNetの最終層(全結合層)を除いたモデルをロード
            self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # 最終の全結合層を恒等関数に置き換え、その直前の2048次元の特徴量ベクトルを出力
            self.base_model.fc = nn.Identity()
        def forward(self, x):
            x = self.base_model(x)
            return x
            
    # 画像の前処理
    preprocess = transforms.Compose([
        # PIL画像をPyTorchテンソルに変換 (値の範囲が[0, 255]から[0.0, 1.0]に変換される)
        transforms.ToTensor(),
        # 短辺が256ピクセルになるようにリサイズ
        transforms.Resize(256, antialias=True),
        # 中央部分の224x224ピクセルを切り出す
        transforms.CenterCrop(224),
        # ImageNetの平均と標準偏差で正規化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. ネットワークをインスタンス化し、評価モードに設定(もし添付ファイルが存在する場合はロード)
    model = FeatureExtractor()
    if os.path.exists(attachment_path):
        model.load_state_dict(torch.load(attachment_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. ミニバッチ処理の準備
    num_images = len(input_data)
    all_features = []  # 各バッチの結果を格納するリスト

    # 3. 特徴量の抽出（ミニバッチループ）
    batch_size = 32
    with torch.no_grad():
        # 指定したbatch_sizeで入力データを分割してループ
        for i in range(0, num_images, batch_size):
            # 現在のミニバッチを取得
            batch_data = input_data[i : i + batch_size]
            
            # preprocessをバッチ内の各画像に適用
            processed_images = [preprocess(Image.fromarray(img)) for img in batch_data]
            
            # 前処理済みのテンソルのリストを1つのバッチテンソルにスタック
            processed_tensor = torch.stack(processed_images).to(device)

            # ミニバッチの特徴量を抽出
            batch_features = model(processed_tensor)
            
            # 結果をCPUに移してリストに保存
            all_features.append(batch_features.cpu())

    # 4. 全てのミニバッチの結果を結合
    final_features = torch.cat(all_features, dim=0)

    return final_features.numpy()


FILEPATH_INPUT_DATA_JSON = r"train/dataset.json"
ANSWER_VALUE_TYPE = "FeatureExtraction"
MULTI_DATA = True
INPUT_DATA_TYPE = "image-3ch"

def read_dataset(path_json, answer_value_type=int, multi_data=False, input_data_type="image-3ch"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_open = open(os.path.join(base_dir, path_json), 'r')
    dataset = json.load(json_open)

    filename_list = []
    input_data_list = [] #入力データ
    correct_list = [] #正解値
    parameter_list = [] #パラメータ
    num_problem = 0

    for item in dataset["data"]:
        try:
            # 正解値
            if type(answer_value_type) is str:
                if answer_value_type == "FeatureExtraction":
                    # テストデータのラベルList
                    correct_list.append(item["gt"])
                else:
                    raise(ValueError(f"answer_value_typeの指定({answer_value_type})が不正です。"))
            else:
                raise(ValueError(f"answer_value_typeの指定({answer_value_type})が不正です。"))

            # 入力データ
            data = []
            filename = []
            if multi_data:
                for path in item["path"]:
                    # 画像読み込み
                    filename.append(path)
                    img = np.array(Image.open(os.path.join(os.path.dirname(path_json), path)))
                    if input_data_type == "image-3ch":
                        # 入力画像が2次元（グレースケール）の場合、3次元に拡張する
                        if img.ndim == 2:
                            # 2次元配列を3回重ねて3次元配列にする (例: (H, W) -> (H, W, 3))
                            img = np.stack([img] * 3, axis=-1)
                    data.append(img)
            else:
                raise(ValueError("multi_dataがFalseです。"))

            input_data_list.append(data)
            filename_list.append(filename)

            # Feature Extraction Taskの場合、shot数をパラメータとして読み込む
            if answer_value_type == "FeatureExtraction":
                parameter = [item["shots"], item["try"]] #[shot数, 試行数]
                parameter_list.append(parameter)

            num_problem += 1
        except Exception as e:
            print(f"入力データ({num_problem})の読み込みに失敗しました：{e}")
            continue

    return num_problem, filename_list, input_data_list, parameter_list, correct_list


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


def print_evaluation_details(detail: dict):
    """
    評価結果の詳細（detail）を整形して表示する関数。
    最新の 'evaluateFeatureExtraction' の出力形式に対応。
    GT数は整数で表示する。

    Args:
        detail (dict): evaluateFeatureExtractionから返される詳細評価結果の辞書。
                       'gt', 'tp', 'fn', 'fp' キーを含むことを想定。
    """
    if not detail:
        print("詳細な評価結果はありません。")
        return

    print("--- クラス別の詳細評価結果 ---")
    
    # ヘッダーを定義（各列の幅を調整）
    header = (
        f"{'Class':<6} | {'GT':<6} | {'TP':<6} | {'FN':<6} | {'FP':<6} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
    )
    print(header)
    print("-" * len(header))  # ヘッダーの下線

    # 各クラスの結果をループして表示
    for class_id, metrics in detail.items():
        # .get(key, 0.0)でキーが存在しない場合も安全に処理
        gt = metrics.get('gt', 0.0)
        tp = metrics.get('tp', 0.0)
        fn = metrics.get('fn', 0.0)
        fp = metrics.get('fp', 0.0)
        p = metrics.get('precision', 0.0)
        r = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        
        # GTは整数（小数点以下0桁）、TP/FN/FPは小数点以下2桁、その他は4桁でフォーマット
        row = (
            f"{str(class_id):<6} | {gt:<6.0f} | {tp:<6.1f} | {fn:<6.1f} | {fp:<6.1f} | "
            f"{p:<10.4f} | {r:<10.4f} | {f1:<10.4f}"
        )
        print(row)
    
    print("-" * len(header))
    

def main():
    # カレントディレクトリ設定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # データ読み込み。224x224に切り抜かれた画像データ。
    print(f"Loading dataset...")
    num_problem, filename_list, input_data_list, parameter_list, correct_list = read_dataset(
        FILEPATH_INPUT_DATA_JSON, ANSWER_VALUE_TYPE, MULTI_DATA, INPUT_DATA_TYPE)

    # ユーザ作成の処理を実行して、特徴量を算出
    answer_list = []
    for i in range(num_problem):
        print(f"Resolve {i+1} / {num_problem}")
        answer_list.append(recognition(input_data_list[i], "feature_extractor_weights.pth"))

    # 評価
    average_f1_scores = []
    for i in range(num_problem):
        print(f"Evaluate {i+1} / {num_problem}")
        num_shots, num_try = parameter_list[i]
        average_f1_score, detail = evaluateFeatureExtraction(answer_list[i], correct_list[i], num_shots, num_try)
        average_f1_scores.append(average_f1_score)
        print(f"Average F1 score for problem {i+1}: {average_f1_score:.4f}")
        print_evaluation_details(detail)

    print(f"Mean F1 score: {np.average(np.array(average_f1_scores)):.4f}")


if __name__ == "__main__":
    main()
