import os
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score


def recognition(input_data: np.ndarray, attachment_path: str) -> np.ndarray:
    # 関数名、引数、戻り値の定義は変更しないでください。
    # input_data: 入力画像のnumpy配列 (画像数, 縦, 横, 3ch)
    # attachment_path: 添付アップロードされたファイルのパス

    # ResNetをベースとした特徴抽出ネットワークを定義
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            # ResNetの最終層を除いたモデルをロード
            self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.base_model.fc = nn.Identity()
            
            # 2048次元を256次元に削減する平均プーリング層
            self.pool_layer = nn.AvgPool1d(kernel_size=8, stride=8)

        def forward(self, x):
            # ネットワークの入力から出力までの一連の流れを定義
            x = self.base_model(x)
            x = x.unsqueeze(1)       # プーリング層のため次元追加 [N, 1, 2048]
            x = self.pool_layer(x)
            x = x.squeeze(1)         # 不要な次元を削除 [N, 256]
            return x

    # 1. ネットワークをインスタンス化し、評価モードに設定(もし添付ファイルが存在する場合はロード)
    model = FeatureExtractor()
    if os.path.exists(attachment_path):
        model.load_state_dict(torch.load(attachment_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. ミニバッチ処理の準備
    num_images = input_data.shape[0]
    all_features = []  # 各バッチの結果を格納するリスト
    # 前処理はループの外で一度定義
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 3. 特徴量の抽出（ミニバッチループ）
    batch_size = 32
    with torch.no_grad():
        # 指定したbatch_sizeで入力データを分割してループ
        for i in range(0, num_images, batch_size):
            # 現在のミニバッチを取得
            batch_data = input_data[i : i + batch_size]
            
            # ミニバッチの前処理
            input_tensor = torch.from_numpy(batch_data).float().permute(0, 3, 1, 2) / 255.0
            processed_tensor = normalize(input_tensor).to(device)

            # ミニバッチの特徴量を抽出
            batch_features = model(processed_tensor)
            
            # 結果をCPUに移してリストに保存 (GPUメモリを効率的に使うため)
            all_features.append(batch_features.cpu())

    # 4. 全てのミニバッチの結果を結合
    final_features = torch.cat(all_features, dim=0)

    return final_features.numpy()


FILEPATH_INPUT_DATA_JSON = r"train/dataset.json"
ANSWER_VALUE_TYPE = "FeatureExtraction"
MULTI_DATA = True
INPUT_DATA_TYPE = "image-3ch"

# 画像の中心を正方形で切り抜き、指定サイズに満たない場合はグレーで埋める。
def center_crop(img, crop_size):
    # 入力画像が2次元（グレースケール）の場合、3次元に拡張する
    if img.ndim == 2:
        # 2次元配列を3回重ねて3次元配列にする (例: (H, W) -> (H, W, 3))
        img = np.stack([img] * 3, axis=-1)

    # 画像の高さと幅を取得
    h, w, _ = img.shape

    # 新しい画像のサイズ
    output_h = crop_size
    output_w = crop_size

    # 新しい画像を作成 (グレーの輝度値127で初期化)
    # dtypeをuint8にすることで、OpenCVの画像として扱いやすくする
    new_img = np.full((output_h, output_w, 3), 127, dtype=np.uint8)

    # 切り抜く領域の座標を計算
    # 画像の中心
    center_y, center_x = h // 2, w // 2

    # 切り抜き領域の左上と右下の座標
    start_y = center_y - crop_size // 2
    end_y = start_y + crop_size
    start_x = center_x - crop_size // 2
    end_x = start_x + crop_size

    # 元の画像から切り抜ける範囲を計算
    # 切り抜き範囲が画像からはみ出す場合に対応
    # y軸方向
    crop_start_y = max(0, start_y)
    crop_end_y = min(h, end_y)
    
    # x軸方向
    crop_start_x = max(0, start_x)
    crop_end_x = min(w, end_x)

    # 新しい画像に貼り付ける領域の座標を計算
    paste_start_y = max(0, -start_y)
    paste_end_y = paste_start_y + (crop_end_y - crop_start_y)
    paste_start_x = max(0, -start_x)
    paste_end_x = paste_start_x + (crop_end_x - crop_start_x)

    # 実際に切り抜いて貼り付け
    cropped_part = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
    new_img[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = cropped_part

    return new_img


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
                    img = center_crop(img, 224)  # 画像を中心で切り抜き
                    data.append(img)
            else:
                raise(ValueError("multi_dataがFalseです。"))

            input_data = np.array(data, dtype=data[0].dtype)
            input_data_list.append(input_data)
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


def evaluateFeatureExtraction(features: np.ndarray, corrects: np.ndarray, num_shots: int, num_try: int) -> float:
    """
    特徴量の識別性能をN-shotの最近傍法で評価し、マクロ平均F1スコアを返す。
    num_try回試行し、その平均F1スコアを算出する。

    Args:
        features (np.ndarray): 特徴量集合 (画像数 x 特徴次元数)。
        corrects (np.ndarray): 各特徴量に対応する正解クラスラベル (画像数,)。
        num_shots (int): 各クラスから登録データとしてランダムに選択するサンプル数。
        num_try (int): 評価の試行回数。シードを0からnum_try-1まで変えて実行する。

    Returns:
        float: num_try回試行したF1スコアの算術平均。
    """
    
    # 各試行のF1スコアを格納するリスト
    f1_scores = []

    # num_tryの回数だけ評価を繰り返す
    for i in range(num_try):
        np.random.seed(i)

        # Step 1: 登録データ(gallery)を作成する
        # ----------------------------------------------------------------------
        gallery_features_list = []
        gallery_labels_list = []
        unique_classes = np.unique(corrects)

        for class_id in unique_classes:
            class_indices = np.where(corrects == class_id)[0]
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
            continue

        gallery_features = np.vstack(gallery_features_list)
        gallery_labels = np.array(gallery_labels_list)

        # Step 2: 識別処理 (最近傍法 + コサイン類似度)
        # ----------------------------------------------------------------------
        similarities = cosine_similarity(features, gallery_features)
        nearest_indices = np.argmax(similarities, axis=1)
        predicted_labels = gallery_labels[nearest_indices]

        # Step 3: 評価 (F1スコアの算出)
        # ----------------------------------------------------------------------
        trial_f1_score = f1_score(y_true=corrects, y_pred=predicted_labels, average='macro', zero_division=0)
        
        # 計算したスコアをリストに追加
        f1_scores.append(trial_f1_score)
    
    # 試行が一度も実行されなかった場合は0.0を返す
    if not f1_scores:
        return 0.0
    
    # リストに格納された全スコアの平均値を計算
    average_f1 = np.mean(f1_scores)
    
    return average_f1


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
        average_f1_score = evaluateFeatureExtraction(answer_list[i], correct_list[i], num_shots, num_try)
        print(f"Average F1 score for problem {i+1}: {average_f1_score:.4f}")
        average_f1_scores.append(average_f1_score)
    print(f"Mean F1 score: {np.average(np.array(average_f1_scores)):.4f}")


if __name__ == "__main__":
    main()
