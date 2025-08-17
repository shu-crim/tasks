import numpy as np


class ImageProc:
    # 画像の中心を正方形で切り抜き、指定サイズに満たない場合はグレーで埋める。
    @staticmethod
    def center_crop(img, crop_size=224):
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