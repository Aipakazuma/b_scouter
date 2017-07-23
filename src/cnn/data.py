from PIL import Image
from tqdm import tqdm
import numpy as np
import os


# TODO: ラベル付をちゃんとしたいので、恐らくcsvを読み込むことになるはず
# TODO: テストデータの用意
class Data():
    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.data_sets = []
        self._read()


    def _read(self):
        for root, dirs, files in os.walk(self.data_dir_path):
            for file in tqdm(files):
                self._add_image(root, file)

        self.data_sets = np.asarray(self.data_sets, dtype=np.float32)


    def _add_image(self, root, file):
        with Image.open(os.path.join(root, file)) as image_file:
            # サイズが違う場合がある
            if image_file.size != (120, 169):
                image_file = image_file.resize((120, 169))

            # channel数が違う場合もある
            if image_file.mode != 'RGB':
                rgb_img = Image.new('RGB', image_file.size)
                rgb_img.paste(image_file)
                image_file = rgb_img

            image_file = np.asarray(image_file)
            self.data_sets.append(image_file)
