#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/SQL_UTILS.py
#    Description:    常见数据库算法
#    Author:     WY
#    Date: 2020/05/26
#    LastEditTime: 2020/07/01
# -------------------------------------------------


class PostGre:
    """
    Postgreku库连接封装类，默认host="172.16.3.250",user="postgres",pwd="1234567",db="nlp",port=5432
    传递参数可改变默认值

    postgre=PostGre(host="172.16.3.250", user="postgres", pwd="1234567", db="nlp")
    """
    def __init__(self,
                 host="172.16.3.250",
                 user="postgres",
                 pwd="1234567",
                 db="nlp",
                 port=5432):
        self.host = host
        self.db = db
        self.user = user
        self.pwd = pwd
        self.port = port
        self._conn = self._connect()
        self._cursor = self._conn.cursor()

    def _connect(self):
        import psycopg2
        try:
            con = psycopg2.connect(database=self.db,
                                   user=self.user,
                                   password=self.pwd,
                                   host=self.host,
                                   port=self.port)
        except Exception as e:
            print("连接失败检查连接文件", e)
        return con

    def select(self, sqlCode):
        """
        默认输入SQl直接查询
        """
        try:
            self._cursor.execute(sqlCode)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sqlCode)
        self._conn.commit()
        result = self._cursor.fetchall()
        return result

    def insert(self, sqlCode):
        self.common(sqlCode)

    """
    使用executemany完成数据的插入和更新
    'INSERT INTO user values(%s,%s,%s)'后接的table是和values里面内容对应
    mysql = postgre()
    table=[[1,"zhangsan"],[2,"zhangsan2"]]
    sql = "INSERT INTO lrd values(%s,%s)"
    mysql.executemany(sql,table)
    """

    def executemany(self, sqlCode, table):
        try:
            self._cursor.executemany(sqlCode, table)
        except Exception as e:
            print(e)
            self._conn.rollback()
            # self._cursor.executemany(sqlCode,table)
        self._conn.commit()

    def update(self, sqlCode):
        self.common(sqlCode)

    def delete(self, sqlCode):
        self.common(sqlCode)

    def close(self):
        self._cursor.close()
        self._conn.close()

    def insertAndGetField(self, sql_code, field):
        """
        插入数据，并返回当前 field
        :param sql_code:
        :param field:
        :return:
        """
        try:
            self._cursor.execute(sql_code + " RETURNING " + field)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sql_code + " RETURNING " + field)
        self._conn.commit()

        return self._cursor.fetchone()

    def common(self, sqlCode):
        try:
            self._cursor.execute(sqlCode)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sqlCode)
        self._conn.commit()


class MYSQL:
    """
        MYSQL库连接封装类，默认host="172.16.3.204", user="root", pwd="root", db="nlp", port=3306
        传递参数可改变默认值

        mysql = MYSQL(host="172.16.3.204", user="root", pwd="root", db="nlp")
    """
    def __init__(self,
                 host="172.16.3.204",
                 user="root",
                 pwd="root",
                 db="nlp",
                 port=3306):
        self.host = host
        self.db = db
        self.user = user
        self.pwd = pwd
        self.port = port
        self._conn = self._connect()
        self._cursor = self._conn.cursor()

    def _connect(self):
        import pymysql
        try:
            con = pymysql.connect(host=self.host,
                                  user=self.user,
                                  password=self.pwd,
                                  database=self.db,
                                  port=self.port)
        except Exception as e:
            print("连接失败检查连接文件", e)
        return con

    def select(self, sqlCode):
        """
        默认输入SQl直接查询
        """
        try:
            self._cursor.execute(sqlCode)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sqlCode)
        self._conn.commit()
        result = self._cursor.fetchall()
        return result

    def insert(self, sqlCode):
        self.common(sqlCode)

    """
    使用executemany完成数据的插入和更新
    'INSERT INTO user values(%s,%s,%s)'后接的table是和values里面内容对应
    mysql = postgre()
    table=[[1,"zhangsan"],[2,"zhangsan2"]]
    sql = "INSERT INTO lrd values(%s,%s)"
    mysql.executemany(sql,table)
    """

    def executemany(self, sqlCode, table):
        try:
            self._cursor.executemany(sqlCode, table)
        except Exception as e:
            print(e)
            self._conn.rollback()
            # self._cursor.executemany(sqlCode,table)
        self._conn.commit()

    def update(self, sqlCode):
        self.common(sqlCode)

    def delete(self, sqlCode):
        self.common(sqlCode)

    def close(self):
        self._cursor.close()
        self._conn.close()

    def common(self, sqlCode):
        try:
            self._cursor.execute(sqlCode)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sqlCode)
        self._conn.commit()


class MSSQL:
    """
    对pymssql的简单封装
    pymssql库，该库到这里下载：http://www.lfd.uci.edu/~gohlke/pythonlibs/#pymssql
    使用该库时，需要在Sql Server Configuration Manager里面将TCP/IP协议开启
    用法：
    mssql = MSSQL(host="192.168.9.174", user="ias", pwd="ias", db="master", port=3306)
    """
    def __init__(self,
                 host="192.168.9.174",
                 user="ias",
                 pwd="1234567",
                 db="master",
                 port=3306):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db
        self.port = port
        self._conn = self.__GetConnect()
        self._cursor = self._conn.cursor()

    def __GetConnect(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        import pymssql

        try:
            conn = pymssql.connect(host=self.host,
                                   user=self.user,
                                   password=self.pwd,
                                   database=self.db,
                                   port=self.port,
                                   charset="utf8")
        except Exception as e:
            print("连接失败检查连接文件", e)
        return conn

    def select(self, sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段

        调用示例：
                ms = MSSQL(host="localhost",user="sa",pwd="123456",db="PythonWeiboStatistics")
                resList = ms.ExecQuery("SELECT id,NickName FROM WeiBoUser")
                for (id,NickName) in resList:
                    print str(id),NickName
        """
        self._cursor.execute(sql)
        resList = self._cursor.fetchall()
        return resList

    def insert(self, sqlCode):
        self.common(sqlCode)

    def update(self, sqlCode):
        self.common(sqlCode)

    def delete(self, sqlCode):
        self.common(sqlCode)

    def close(self):
        # 查询完毕后必须关闭连接
        self._cursor.close()
        self._conn.close()

    def common(self, sqlCode):
        try:
            self._cursor.execute(sqlCode)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sqlCode)
        self._conn.commit()


class Oracle:
    '''
        oracle = Oracle(host="192.168.9.222", user="DL", pwd="DL", db="dsecs")
    '''
    def __init__(self,
                 host="192.168.9.222",
                 user="DL",
                 pwd="DL",
                 db="dsecs",
                 port="1521"):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.port = port
        self.database = db

        self._conn = self.__GetConnect()
        self._cursor = self._conn.cursor()

    def __GetConnect(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        import cx_Oracle
        import os
        os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
        try:
            self.oracle = self.user + "/" + self.pwd + "@" + self.host + ":" + self.port + "/" + self.database
            conn = cx_Oracle.connect(self.oracle)
        except cx_Oracle.DatabaseError as msg:
            print("连接失败检查连接文件", msg)
        return conn

    def select(self, sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段
1
        调用示例：
                oracle = Oracle(hostconn="192.168.9.222:1521/dsecs",user="dsecs",pwd="dsecs")
                resList = oracle.select("SELECT * FROM SJD")
                for (id,NickName) in resList:
                    print str(id),NickName
        """
        self._cursor.execute(sql)
        resList = self._cursor.fetchall()
        return resList

    def insert(self, sqlCode):
        self.common(sqlCode)

    """
    mysql = cxOracle()
    table=[[3,"zhangsan"],[4,"zhangsan2"]]
    # sql = "INSERT INTO lrd values(%s,%s)"
    sql = "INSERT INTO LRD values(:1,:2)"
    # sql = "select * FROM YYCDB"
    mysql.executemany(sql,table)
    """

    def executemany(self, sqlCode, table):
        try:
            self._cursor.prepare(sqlCode)
            self._cursor.executemany(None, table)
        except Exception as e:
            print(e)
            self._conn.rollback()
            # self._cursor.executemany(sqlCode,table)
        self._conn.commit()

    def update(self, sqlCode):
        self.common(sqlCode)

    def delete(self, sqlCode):
        self.common(sqlCode)

    def close(self):
        # 查询完毕后必须关闭连接
        self._cursor.close()
        self._conn.close()

    def common(self, sqlCode):
        try:
            self._cursor.execute(sqlCode)
        except Exception as e:
            print(e)
            self._conn.rollback()
            self._cursor.execute(sqlCode)
        self._conn.commit()


class ESwrite:
    """
    ES数据写入库连接封装类，默认host="172.16.3.242",port=9200,Index_name="factjjdb_test2",thread_count=4,chunk_size=500
    传递参数可改变默认值
    """
    def __init__(self,
                 host="172.16.3.242",
                 port=9200,
                 thread_count=4,
                 chunk_size=500):
        self.host = host
        self.port = port
        # self.Index_name = Index_name
        self.thread = thread_count
        self.chunk_size = chunk_size
        self.connes = self._connect()

    def _connect(self):
        from elasticsearch import Elasticsearch
        try:
            con = Elasticsearch([self.host], port=self.port)
        except Exception as e:
            print("连接失败检查连接文件", e)
        return con

    def WriteES(self, Index_name, Tag_name, IndexData, PreResult):
        from elasticsearch import helpers
        """
        ES数据写入库连接封装类，输入数据分别是Index_name是ES数据库中的表名，tag_name为需要打标签的名字，IndexData为需要更新的索引ID编号
        形式如下['J3213225318122300014', 'J3205075218122300001']
        PreResult为预测的二维数组更新内容，没有则新建[[0.93480456],[0.9358239 ],[0.8241926 ],[0.9171963 ]]
        """
        actions = []
        num = len(IndexData)
        for line in range(num):
            # res = str(PreResult[line][0])
            res = round(PreResult[line][0], 3)
            action = {
                '_op_type': 'update',
                "_index": Index_name,
                "_type": "_doc",
                "_id": IndexData[line],
                "doc": {
                    Tag_name: res,
                }
            }
            actions.append(action)
        ess = helpers.parallel_bulk(self.connes, actions, self.thread,
                                    self.chunk_size)
        for ok, response in ess:
            if not ok:
                print(response)


def getParameter(section="PostgreSQL", key="host", file='config.ini'):
    """
    host = getParameter("PostgreSQL", "host")
    user = getParameter("PostgreSQL", "user")
    pwd = getParameter("PostgreSQL", "pwd")
    db = getParameter("PostgreSQL", "db")
    port = int(getParameter("PostgreSQL", "port"))
    **统一返回string形式类型，如需转换请自行转换

    :param section:
    :param value:
    :param file:
    :return:
    """
    import configparser
    # 创建配置文件对象
    con = configparser.RawConfigParser()
    # 读取文件
    con.read(file, encoding='utf-8')
    # # 获取所有section
    # sections = con.sections()
    # print("当前文件下包含的所有配置信息名称为：",sections)
    # 获取特定配置信息
    items = con.items(section)  # 返回结果为元组
    # print(items)
    items = dict(items)
    return items[key]


def setParameter(section="PostgreSQL",
                 key="host",
                 value="172.16.3.220",
                 file='config.ini'):
    """
    host = setParameter("PostgreSQL", "host","172.16.3.220")
    :param section:
    :param value:
    :param file:
    :return:
    """
    import configparser
    # 创建配置文件对象
    con = configparser.RawConfigParser()
    # 读取文件
    con.read(file, encoding='utf-8')
    # 判断section是否存在
    if not con.has_section(section):
        con.add_section(section)
    con.set(section, key, value)
    con.write(open(file, "w"))
    if getParameter(section=section, key=key, file=file):
        return True
    else:
        return False


if __name__ == '__main__':
    pass
