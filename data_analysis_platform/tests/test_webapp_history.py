import io
import json
from data_analysis_platform.webapp.app import app
from data_analysis_platform import config
from data_analysis_platform.utils import session_store


def test_analyze_creates_session_history(tmp_path, monkeypatch):
    history_file = tmp_path / 'session_history.json'
    monkeypatch.setattr(config, 'SESSION_STORE', str(history_file))

    client = app.test_client()
    data = b"col1,col2\n1,2\n3,4\n"

    rv = client.post(
        '/analyze',
        data={
            'file': (io.BytesIO(data), 'test.csv'),
            'analysis_type': '趋势',
            'chart_type': 'line',
            'format': 'html',
        },
        content_type='multipart/form-data',
        follow_redirects=True,
    )

    assert rv.status_code == 200
    body = rv.get_data(as_text=True)
    assert '分析结果' in body
    assert history_file.exists()

    sessions = session_store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]['dataset'] == 'test.csv'
    assert '趋势' in sessions[0]['query']

    rv2 = client.get('/history')
    assert rv2.status_code == 200
    assert '会话历史' in rv2.get_data(as_text=True)
    assert 'test.csv' in rv2.get_data(as_text=True)
