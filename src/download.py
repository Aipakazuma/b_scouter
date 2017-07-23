from tqdm import tqdm
import requests
import os
import sys
import argparse

def download_image(url, timeout = 10):
    """画像をダウンロードする."""
    response = requests.get(url, allow_redirects=False, timeout=timeout)
    if response.status_code != 200:
        e = Exception('HTTP status: ' + response.status_code)
        raise e

    content_type = response.headers['content-type']
    if 'image' not in content_type:
        e = Exception('Content-Type: ' + content_type)
        raise e

    return response.content


def save_image(filename, image):
    """画像を保存する."""
    with open(filename, 'wb') as fout:
        fout.write(image)


def argument():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv', action='store', type=str, help='parser csv.')
    parser.add_argument('--images_path', action='store', type=str, help='save images path.')
    return parser.parse_args()

if __name__ == '__main__':
    """メイン."""
    args = argument()
    idx = 0

    with open(args.csv, 'r') as fin:
        for line in fin:
            line_strip = line.split(',')
            id_name = line_strip[0]
            url = line_strip[1]
            filename = '%d.jpg' % (int(id_name))

            print('%s' % (url))
            try:
                image = download_image(url)
                save_image(filename, image)
                idx += 1
            except KeyboardInterrupt:
                break
            except Exception as err:
                print('%s' % (err))
