# B Scouter

完璧ネタです。夢です。ロマンです（多分）。



# Usage

まだない



# Todo

言語は全部Python（多分）

* Tensorflowの使い方勉強
    * ちょっと覚えた
* TensorflowでCNNを試す
    * cfar-10のチュートリアルでいいんじゃないかな？
* Tensorflowのチューニング
* AWSでインフラ構築
* APIは用意しない
  * 多分レスポンスがあかんことになる
  * 学習済みのモデルをラズパイに配置すればおｋ
  * とわいえ、人のデータも見てみたい
    * (時間があれば)TensorflowでAPIを作成する
        * デモサイトを用意
            * Webサービスで公開
            * Flaskでやります
* OpenCVでDVDパッケージを検知する
    * 多分使うやつ
    * キャリブレーション（場所によっては真四角じゃないはず）
        * DVDのサイズ(W136xD14xH190)から対比(68:95, 42%:58%)ということがわかっている
        * 上のサイズはアマレータイプと呼ばれるものらしい。んで映画で使われるとのことなのでこれで
    * アフィン変換（場所によっては歪んでいるはず）
    * rect（囲んで切り出しするから）
        * dilate(original_image) -> diff = substract(dilate_image - original_image) -> 255 - diff で線画を取れることがわかった
            * このあとに2値化（パラメータになりそう）して、rectすればdetectがうまくいきそう
            * うまくとれないなら、手動で操作してもよさそう
                * 本当はボタンがほしいけど、iphoneから操作しようかな（vncとかsshができるっぽい）
* 訓練データをクロール（GEOから）
    * 今やっている
* 機材購入
    * [ラズパイ](http://jp.rs-online.com/web/p/processor-microcontroller-development-kits/1225826/)
    * [バッテリー](https://www.amazon.co.jp/Anker-PowerCore-%E3%83%A2%E3%83%90%E3%82%A4%E3%83%AB%E3%83%90%E3%83%83%E3%83%86%E3%83%AA%E3%83%BC-2016%E5%B9%B48%E6%9C%88%E6%9C%AB%E6%99%82%E7%82%B9-A1263011/dp/B019GNUT0C)
    * [SDカード](http://jp.rs-online.com/web/p/secure-digital-cards/1213897/?origin=PSF_437585|acc)
    * [ディスプレイ](http://jp.rs-online.com/web/p/lcd-colour-displays/9094105/)
    * [カメラ](http://jp.rs-online.com/web/p/products/9132664/?grossPrice=Y&cm_mmc=JP-PLA-_-google-_-PLA_JP_JP_%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF_%2F%E5%91%A8%E8%BE%BA%E6%A9%9F%E5%99%A8-_-&mkwid=sqfLah1Zf-dc|pcrid|198830549484|pkw||pmt||prd|9132664)

勉強した内容は細かくでいいからブログに反映 => 会社のブログに書くことになった



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
