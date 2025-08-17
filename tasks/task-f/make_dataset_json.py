import os
import json
import random

# --- 設定項目 ---

# 1. Caltech-101の '101_ObjectCategories' フォルダへのパスを指定
DATASET_ROOT_PATH = r"101_ObjectCategories"

# 2. 選択するクラスの数
NUM_CLASSES_TO_SELECT = 80

# 3. 各クラスから選択する画像の数
NUM_IMAGES_PER_CLASS = 150

# 4. 出力するJSONファイル名
OUTPUT_JSON_PATH = "caltech101_dataset.json"


def create_dataset_json(dataset_root_path: str, num_classes: int, num_images_per: int, output_path: str):
    """
    ローカルのCaltech-101データセットからランダムにデータを選択し、JSONファイルを作成する関数。
    gtリストは改行せずに一行で出力する。

    Args:
        dataset_root_path (str): '101_ObjectCategories' フォルダへのパス。
        num_classes (int): 選択するクラスの数。
        num_images_per (int): 各クラスから選択する画像の数。
        output_path (str): 出力するJSONファイル名。
    """
    if not os.path.isdir(dataset_root_path):
        print(f"エラー: 指定されたパス '{dataset_root_path}' が見つからないか、ディレクトリではありません。")
        print("DATASET_ROOT_PATH を正しい '101_ObjectCategories' フォルダのパスに設定してください。")
        return

    # 'BACKGROUND_Google' を除外したクラス（ディレクトリ）のリストを取得
    try:
        all_classes = [d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d)) and d != 'BACKGROUND_Google']
    except Exception as e:
        print(f"エラー: ディレクトリの読み込み中にエラーが発生しました: {e}")
        return

    if len(all_classes) < num_classes:
        print(f"エラー: 要求されたクラス数 ({num_classes}) が、利用可能なクラス数 ({len(all_classes)}) を超えています。")
        return

    # 指定された数のクラスをランダムに選択
    selected_classes = random.sample(all_classes, num_classes)
    print(f"選択されたクラス: {selected_classes}")

    # クラス名とユニークなID（0から始まる整数）をマッピング
    class_to_id = {class_name: i for i, class_name in enumerate(selected_classes)}

    file_paths = []
    ground_truths = []

    # JSONに保存するパスのトップレベルフォルダ名を取得 (例: "101_ObjectCategories")
    base_folder_name = os.path.basename(os.path.normpath(dataset_root_path))

    # 選択された各クラスについて処理
    for class_name in selected_classes:
        class_id = class_to_id[class_name]
        class_dir = os.path.join(dataset_root_path, class_name)
        images_in_class = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]

        if len(images_in_class) < num_images_per:
            print(f"警告: クラス '{class_name}' の画像は {len(images_in_class)} 枚しかありません（要求: {num_images_per}）。存在する全てを選択します。")
            images_to_sample = images_in_class
        else:
            images_to_sample = random.sample(images_in_class, num_images_per)

        for image_name in images_to_sample:
            relative_path = os.path.join(base_folder_name, class_name, image_name).replace("\\", "/")
            file_paths.append(relative_path)
            ground_truths.append(class_id)
    
    # "path"リスト部分の文字列を作成（各要素を改行・インデント）
    # リストの各パスをダブルクォートで囲み、カンマと改行で連結する
    path_list_str = ",\n".join([f'        "{p}"' for p in file_paths])
    
    # "gt"リスト部分の文字列を作成（改行なし）
    # json.dumpsをインデントなしで使うと、コンパクトな一行のリスト文字列が生成される
    gt_list_str = json.dumps(ground_truths)

    # 最終的なJSON文字列をf-stringで組み立てる
    final_json_string = f'''{{
    "path": [
{path_list_str}
    ],
    "gt": {gt_list_str}
}}'''

    # 組み立てた文字列をファイルに書き込む
    with open(output_path, 'w') as f:
        f.write(final_json_string)

    print(f"\nJSONファイル '{output_path}' が正常に作成されました。")
    print(f"合計 {len(file_paths)} 件のデータが含まれています。")


if __name__ == "__main__":
    # JSONファイルの生成を実行
    create_dataset_json(
        dataset_root_path=DATASET_ROOT_PATH,
        num_classes=NUM_CLASSES_TO_SELECT,
        num_images_per=NUM_IMAGES_PER_CLASS,
        output_path=OUTPUT_JSON_PATH
    )
