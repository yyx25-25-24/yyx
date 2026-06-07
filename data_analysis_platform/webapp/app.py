from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, session
import os
import sys
import importlib
import pandas as pd
from werkzeug.utils import secure_filename

# ensure workspace root is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# import package config and expose it as top-level `config` so modules using `import config` keep working
pkg_config = importlib.import_module('data_analysis_platform.config')
sys.modules['config'] = pkg_config
# provide shims for imports like `from utils import session_store` used in main.py
try:
    utils_pkg = importlib.import_module('data_analysis_platform.utils')
    sys.modules['utils'] = utils_pkg
    for sub in ('session_store', 'chart_utils', 'report_utils', 'data_utils', 'llm_utils'):
        try:
            sys.modules[f'utils.{sub}'] = importlib.import_module(f'data_analysis_platform.utils.{sub}')
        except Exception:
            pass
except Exception:
    pass

from data_analysis_platform.utils.chart_utils import generate_default_chart
from data_analysis_platform.utils.data_utils import preprocess_df, detect_numeric_columns
from data_analysis_platform.main import local_analysis_engine
import data_analysis_platform.config as config
from data_analysis_platform.utils.report_utils import save_analysis_report
from data_analysis_platform.utils import session_store

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.CHART_SAVE_PATH, exist_ok=True)

ALLOWED_EXT = {'.csv', '.xls', '.xlsx'}


def read_dataframe(save_path: str, filename: str):
    if filename.lower().endswith('.csv'):
        try:
            return pd.read_csv(save_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            return pd.read_csv(save_path, encoding='gbk')
    # Excel file handling
    try:
        return pd.read_excel(save_path, engine='openpyxl')
    except Exception:
        try:
            return pd.read_excel(save_path, engine='xlrd')
        except Exception as exc:
            raise RuntimeError(f'读取 Excel 文件失败: {exc}')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'dev'


def allowed_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXT


def login_required(fn):
    def wrapper(*args, **kwargs):
        # if no credentials set, allow access
        if not config.WEB_USERNAME:
            return fn(*args, **kwargs)
        if session.get('user') == config.WEB_USERNAME:
            return fn(*args, **kwargs)
        return redirect(url_for('login'))
    wrapper.__name__ = fn.__name__
    return wrapper


@app.route('/')
@login_required
def index():
    return render_template('upload.html')


@app.route('/columns', methods=['POST'])
def get_columns():
    file = request.files.get('file')
    if not file or file.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    try:
        df = read_dataframe(save_path, filename)
    except Exception as e:
        return {'error': str(e)}, 400

    # try basic preprocessing
    try:
        df_proc = preprocess_df(df, parse_dates=True, fill_method='ffill')
    except Exception:
        df_proc = df

    cols = list(df_proc.columns)
    numeric_cols = detect_numeric_columns(df_proc)
    categorical = [c for c in cols if c not in numeric_cols]
    dtypes = {c: str(df_proc[c].dtype) for c in cols}
    preview = df_proc.head(5).astype(str).to_dict(orient='records')
    return {
        'columns': cols,
        'numeric': numeric_cols,
        'categorical': categorical,
        'dtypes': dtypes,
        'preview': preview,
    }


@app.route('/login', methods=['GET', 'POST'])
def login():
    if not config.WEB_USERNAME:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
        if user == config.WEB_USERNAME and pwd == config.WEB_PASSWORD:
            session['user'] = user
            return redirect(url_for('index'))
        flash('用户名或密码错误')
    return render_template('login.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory(os.path.abspath(config.CHART_SAVE_PATH), filename)


@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(os.path.abspath(config.REPORT_SAVE_PATH), filename)


@app.route('/history')
@login_required
def history():
    sessions = session_store.list_sessions()
    return render_template('history.html', sessions=sessions)


@app.route('/analyze', methods=['POST'])
def analyze():
    # handle file upload
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('请上传一个文件（CSV 或 Excel）')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        flash('不支持的文件类型')
        return redirect(url_for('index'))

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # read into dataframe
    try:
        df = read_dataframe(save_path, filename)
    except Exception as e:
        flash(f'读取文件失败: {e}')
        return redirect(url_for('index'))

    # preprocess
    try:
        df_proc = preprocess_df(df, parse_dates=True, fill_method='ffill')
    except Exception:
        df_proc = df

    # detect numeric columns and provide better feedback
    numeric_cols = detect_numeric_columns(df_proc)
    cols_info = {c: str(df_proc[c].dtype) for c in df_proc.columns}

    # gather params
    analysis_type = request.form.get('analysis_type', '总体分析')
    chart_type = request.form.get('chart_type', '') or 'line'
    x_col = request.form.get('x_col') or None
    y_col = request.form.get('y_col') or None
    color_scheme = request.form.get('color_scheme') or None
    fmt = request.form.get('format') or config.CHART_FORMAT or 'png'
    width = int(request.form.get('width') or 800)
    height = int(request.form.get('height') or 600)

    # run analysis (use local engine)
    if not numeric_cols:
        columns_text = '\n'.join([f'- {col}: {dtype}' for col, dtype in cols_info.items()])
        report = (
            '当前数据集未识别出数值列。请检查上传的文件是否包含数字，或是否为文本格式。\n'
            '检测到的列和类型如下：\n'
            f'{columns_text}\n'
            '如果你上传的是 Excel 文件，Excel 本身是二进制格式，无法用文本编辑器直接打开。'
        )
    else:
        report = local_analysis_engine(analysis_type, df_proc)

    # generate chart
    chart_path = None
    chart_url = None
    try:
        chart_path = generate_default_chart(df_proc, chart_type, config.CHART_SAVE_PATH, fmt, title=f"{filename} - {chart_type}", x_col=x_col, y_col=y_col, color_scheme=color_scheme, width=width, height=height)
    except Exception as e:
        if fmt in ('png', 'svg') and 'kaleido' in str(e).lower():
            flash('PNG/SVG 导出失败，已回退到 HTML 图表显示。')
            try:
                chart_path = generate_default_chart(df_proc, chart_type, config.CHART_SAVE_PATH, 'html', title=f"{filename} - {chart_type}", x_col=x_col, y_col=y_col, color_scheme=color_scheme, width=width, height=height)
            except Exception as ex2:
                flash(f'HTML 图表也生成失败: {ex2}')
                chart_path = None
        else:
            flash(f'生成图表失败: {e}')
            chart_path = None
    if chart_path:
        chart_name = os.path.basename(chart_path)
        chart_url = url_for('serve_chart', filename=chart_name)

    # generate html report (embedded chart link)
    try:
        report_path = save_analysis_report(analysis_type, (report, [chart_url]) if chart_url else report, filename, config.REPORT_SAVE_PATH, 'html')
        report_name = os.path.basename(report_path)
        report_url = url_for('serve_report', filename=report_name)
    except Exception as e:
        report_url = None
        flash(f'生成报告失败: {e}')

    # persist session
    try:
        session_store.append_session({
            'user': session.get('user'),
            'dataset': filename,
            'query': analysis_type,
            'result_preview': str(report)[:300],
            'charts': [chart_url] if chart_url else [],
        })
    except Exception:
        pass

    return render_template('result.html', report=report, chart_path=chart_url, report_path=report_url, upload_name=filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
