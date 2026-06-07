import io
import os
from data_analysis_platform.webapp.app import app
from data_analysis_platform import config


def test_columns_and_analyze():
    client = app.test_client()
    # create a small csv
    data = b"col1,col2\n1,2\n3,4\n"
    rv = client.post('/columns', data={'file': (io.BytesIO(data), 'test.csv')}, content_type='multipart/form-data')
    assert rv.status_code == 200
    j = rv.get_json()
    assert 'columns' in j and 'col1' in j['columns']

    # analyze
    rv2 = client.post('/analyze', data={'file': (io.BytesIO(data), 'test.csv'), 'analysis_type': '趋势', 'chart_type': 'line', 'format': 'html'}, content_type='multipart/form-data', follow_redirects=True)
    assert rv2.status_code == 200
    body = rv2.get_data(as_text=True)
    assert '分析结果' in body or '分析结果' in body