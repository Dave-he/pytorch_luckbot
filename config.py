import configparser

# 全局变量
config = None


def load_config(file_path='config.cfg'):
    """
    读取配置文件并存储在全局变量中
    :param file_path: 配置文件路径
    """
    global config
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(file_path)


def get_config(section='DEFAULT', option=''):
    """
      获取配置项
      :param section: 配置部分
      :param option: 配置选项
      :return: 配置值
       """
    if config is None:
        raise ValueError("配置文件未加载，请先调用 load_config()")
    return config.get(section, option)


def get_config_int(section='DEFAULT', option=''):
    """
    获取配置项（整数）
    :param section: 配置部分
    :param option: 配置选项
    :return: 配置值
    """
    if config is None:
        raise ValueError("配置文件未加载，请先调用 load_config()")
    return config.getint(section, option)


def get_config_boolean(section='DEFAULT', option=''):
    """
     获取配置项（布尔值）
     :param section: 配置部分
     :param option: 配置选项
     :return: 配置值
    """
    if config is None:
        raise ValueError("配置文件未加载，请先调用 load_config()")
    return config.getboolean(section, option)


def get_config_float(section='DEFAULT', option=''):
    """
       获取配置项（浮点数）
       :param section: 配置部分
       :param option: 配置选项
       :return: 配置值
     """
    if config is None:
        raise ValueError("配置文件未加载，请先调用 load_config()")
    return config.getfloat(section, option)
