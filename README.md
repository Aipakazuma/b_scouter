# B Scouter

完璧ネタです。夢です。ロマンです（多分）。



# Usage

まだない



# Todo

言語は全部Python（多分）

* Tensorflowの使い方勉強
* TensorflowでCNNを試す
    * cfar-10のチュートリアルでいいんじゃないかな？
* Tensorflowのチューニング
* AWSでインフラ構築
* TensorflowでAPIを作成する
    * Webサービスで公開
    * Flaskでやります
* OpenCVでDVDパッケージを検知する
    * 多分使うやつ
    * キャリブレーション（場所によっては真四角じゃないはず）
        * DVDのサイズ(W136xD14xH190)から対比(68:95, 42%:58%)ということがわかっている
        * 上のサイズはアマレータイプと呼ばれるものらしい。んで映画で使われるとのことなのでこれで
    * アフィン変換（場所によっては歪んでいるはず）
    * rect（囲んで切り出しするから）
* 訓練データをクロール（GEOから）

勉強した内容は細かくでいいからブログに反映



# If I have time

* ラズパイとかでIoT化してみたい 
    * 2万あればいけそう


# could do

## ubuntuをラズパイにインストール

```sh
$ xzcat ~/Downloads/ubuntu-16.04-preinstalled-server-armhf+raspi3.img.xz | dd of=/dev/disk2s1
dd: /dev/disk2s1: Permission denied
$ xzcat ~/Downloads/ubuntu-16.04-preinstalled-server-armhf+raspi3.img.xz | sudo dd of=/dev/disk2s1
Password:
dd: /dev/disk2s1: Resource busy  # !??
# 自動でマウントしていたのでアンマウント
$ sudo diskutil umount "/Volumes/NO NAME"
$ xzcat ~/Downloads/ubuntu-16.04-preinstalled-server-armhf+raspi3.img.xz | sudo dd of=/dev/disk2s1
```

* [第450回　Raspberry Pi 3にUbuntu 16.04 LTSをインストールする（Ubuntuカーネル編）](http://gihyo.jp/admin/serial/01/ubuntu-recipe/0450)


結構長い・・・


結局なんどか試したけど、bootできなかった。このOSは諦め、xubuntuを試したらbootできた。

## 画面小さい問題


目が痛い

* [RaspberryPiのHDMIディスプレイ解像度を調整](http://kamuycikap.hatenablog.com/entry/2015/02/19/012349)


ここを参考にしたらできた

```text
hdmi_group=2   # <--- HDMI
hdmi_mode=4
```

これでみえるようになった

## miniconda install

* [Anaconda on Raspberry PI 3](http://qiita.com/jpena930/items/eac02cb4e635bfba83d8)
