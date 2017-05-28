# -*- coding:utf8 -*-

from wsgiref.simple_server import make_server
from cgi import FieldStorage
import base64
import imghdr

html = '''<html>
<head>
    <meta charset="utf-8">
    <title>テスト</title>
</head>
<body>
    <h1>テスト</h1>
    <form action="/" method="post" enctype="multipart/form-data">
        <p>
            <label for="image">画像アップロード</label>
            <input type="file" name="image">
        </p>
        <p>
            <input type="submit" name="upload image">
        </p>
    </form>
    <img src="data:image/png;base64,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">
</body>
</html>'''.encode('utf-8')


def post_image(environ):
    form = FieldStorage(fp=environ['wsgi.input'], environ=environ, keep_blank_values=True)
    data = {k: form[k].value for k in form}
    base64_image_data = base64.b64encode(data['image']).decode('ascii')

    image_type = imghdr.what(None, data['image'])
    return '<img src="data:image/' + image_type + ';base64,' + base64_image_data + '" alt="テストデータ">'


def response(start_response, body):
    status = '200 OK'
    headers = [
        ('Content-Type', 'text/html; charset=utf-8'),
        ('Content-Length', str(len(body)))
    ]
    start_response(status, headers)
    return [body]


def simple_app(environ, start_response):
    request_method = environ.get('REQUEST_METHOD')
    if 'POST' == request_method:
        # postの処理
        body = post_image(environ=environ).encode('utf-8')
        return response(start_response=start_response, body=body)
    elif 'GET' == request_method:
        # getの処理
        body = html
        return response(start_response=start_response, body=body)
    else:
        pass


httpd = make_server('', 8000, simple_app)
print('Serving on port 8000...')
httpd.serve_forever()
