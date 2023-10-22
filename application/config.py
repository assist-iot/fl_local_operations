import os


def if_env(keyword):
    if keyword in os.environ:
        return os.environ[keyword]
    else:
        return ''


HOST = os.environ['HOST']
PORT = int(os.environ['PORT'])
DB_PORT = int(os.environ['DB_PORT'])
DATA_FORMAT_FILE = os.environ['DATA_FORMAT_FILE']
DATA_PIPELINE_FILE = os.environ['DATA_PIPELINE_FILE']
DATA_FOLDER = os.environ['DATA_FOLDER']
PREPROCESSED_FOLDER = os.environ['PREPROCESSED_FOLDER']
REPOSITORY_ADDRESS = os.environ['REPOSITORY_ADDRESS']
ORCHESTRATOR_SVR_ADDRESS = os.environ['ORCHESTRATOR_SVR_ADDRESS']
ORCHESTRATOR_WS_ADDRESS = os.environ['ORCHESTRATOR_WS_ADDRESS']
WS_TIMEOUT = int(os.environ['WS_TIMEOUT'])
FEDERATED_PORT = int(os.environ['FEDERATED_PORT'])
SERVER_ADDRESS = os.environ['SERVER_ADDRESS']
DATABASE_NAME = os.environ['DATABASE_NAME']
TOTAL_LOCAL_OPERATIONS = os.environ['TOTAL_LOCAL_OPERATIONS']
HM_SECRET_FILE = os.environ['HM_SECRET_FILE']
HM_PUBLIC_FILE = os.environ['HM_PUBLIC_FILE']
