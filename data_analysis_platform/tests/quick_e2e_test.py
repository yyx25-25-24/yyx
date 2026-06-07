import sys
import importlib
from data_analysis_platform import config
# shim top-level config for modules that import `config`
sys.modules['config'] = importlib.import_module('data_analysis_platform.config')
# shim utils package for modules that import `utils`
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
import pandas as pd
from data_analysis_platform.utils.data_utils import preprocess_df
from data_analysis_platform.main import local_analysis_engine
from data_analysis_platform.utils.chart_utils import generate_default_chart
from data_analysis_platform.utils.report_utils import save_analysis_report

fp = config.DATA_FILES[0]
print('using file', fp)
try:
    df = pd.read_excel(fp)
except Exception as e:
    df = pd.read_csv(fp)
print('rows,cols', df.shape)
df_proc = preprocess_df(df, parse_dates=True, fill_method='ffill')
resp = local_analysis_engine('趋势', df_proc)
print('analysis ok')
# ensure at least one numeric column for charting
if not any(df_proc[c].dtype.kind in 'fi' for c in df_proc.columns):
    import numpy as np
    df_proc = pd.DataFrame({'x': list(range(10)), 'y': list(range(10))})

chart = generate_default_chart(df_proc, 'line', config.CHART_SAVE_PATH, 'html', title='test', width=640, height=480)
print('chart saved', chart)
report = save_analysis_report('趋势', (resp, [chart]), 'testdataset', config.REPORT_SAVE_PATH, 'html')
print('report saved', report)
