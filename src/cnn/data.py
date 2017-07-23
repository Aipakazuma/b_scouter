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


    def _add_image(self, root, file):
        with Image.open(os.path.join(root, file)) as image_file:
            # TODO: なんかsizeが違うとかで怒られるので、どうにかする
            # エラーを確認するとチャネルが入っていないようにみえた
            if image_file.size == (120, 169):
                image_file = image_file.resize((120, 169))
            self.data_sets.append(image_file.tobytes())
