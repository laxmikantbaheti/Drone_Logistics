from mlpro.bf.various import Log as MLProLog

class Log(MLProLog):

    C_LOG_LOGI_EVENT = 'Logistic Event'
    C_LOG_LOGI_ERROR = 'Error'

    C_LOG_TYPES = MLProLog.C_LOG_TYPES + [C_LOG_LOGI_ERROR, C_LOG_LOGI_EVENT]

    C_LOG_LEVEL_LOGI = "Logistic"

    C_LOG_LEVELS = MLProLog.C_LOG_LEVELS + [C_LOG_LEVEL_LOGI]

    C_COL_LOGI = '\033[32m'

    def __init__(self):

        MLProLog.__init__(self)