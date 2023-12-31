from flask import Flask, render_template, request, redirect, url_for, make_response, Markup, send_from_directory, send_file
from flask_httpauth import HTTPBasicAuth, HTTPDigestAuth
from werkzeug.utils import secure_filename
import json
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

import os
import glob
import datetime
from enum import Enum
import shutil

from task import Task, Stats, Log


OUTPUT_DIR_NAME = r"output"
UPLOAD_DIR_NAME = r"upload"
ALLOWED_EXTENSIONS = set(['py'])
TASK = {}
SETTING = None
USER_CSV_PATH = r"./data/users.csv"
SETTING_JSON_PATH = r"./data/setting.json"
HASH_METHOD = "pbkdf2:sha256:260000"
USER_MODULE_DIR_NAME = r"user_module"
COOKIE_KEY = "user"
COOKIE_AGE_SEC = 60 * 60 * 24 * 365


class UserData():
    id : str
    email : str
    pass_hash : str
    name : str
    key : str

    def __init__(self, id="", email="", pass_hash="", name="", key="") -> None:
        self.id = id
        self.email = email
        self.pass_hash = pass_hash
        self.name = name
        self.key = key


class Page(Enum):
    HOME = 0,
    TASK = 1,
    BOARD = 2,
    LOG = 3,
    UPLOAD = 4,
    ADMIN = 9


# Flaskオブジェクトの生成
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 #ファイルサイズ制限 2MB
app.config['SECRET_KEY'] = 'secret key here'


def ReadUsersCsv(path:str):
    users = {}
    try:
        with open(path, encoding='utf-8') as f:
            f.readline() # ヘッダを読み飛ばす
            while True:
                line = f.readline()
                if not line:
                    break
                if len(line.rstrip().split(',')) != 5:
                    continue
                email, id, name, key, pass_hash = line.rstrip().split(',')
                users[id] = UserData(id, email, pass_hash, name, key)
    except:
        return {}

    return users


def WriteUsersCsv(path:str, users:dict, must_backup:bool=True) -> bool:
    # バックアップをとる
    if os.path.exists(path):
        try:
            backup_dir = os.path.join(os.path.dirname(path), "backup")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            shutil.copy2(path, os.path.join(backup_dir, datetime.datetime.now().strftime('users_%Y%m%d_%H%M%S_%f.csv')))
        except:
            if must_backup:
                return False
            
    # ディレクトリが無ければ作成
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    # 上書き作成する
    try:
        with open(path, mode='w', encoding='utf-8') as f:
            f.write('email,id,name,key,pass-hash\n')
            for id, user_data in users.items():
                f.write(f"{user_data.email},{id},{user_data.name},{user_data.key},{user_data.pass_hash}\n")
    except:
        return False
    
    return True


def AddUsersCsv(path:str, id:str, email:str, name:str, pass_hash:str, key:str) -> bool:
    # ユーザデータを読み込む
    users = ReadUsersCsv(path)
    if users == None:
        users = {}

    # 重複チェック
    if id in users:
        return False

    # ユーザデータを追加する
    users[id] = UserData(id, email, pass_hash, name, key)

    # 書き込んで結果を返す
    return WriteUsersCsv(path, users, True if os.path.exists(path) else False)


def UpdateUsersCsv(path:str, id:str, target:str, value:str) -> bool:
    users = ReadUsersCsv(path)
    if users == None:
        return False
    if not id in users:
        return False
    
    # 更新する
    if target == 'key':
        users[id].key = value
    elif target == 'name':
        users[id].name = value
    elif target == 'pass_hash':
        users[id].pass_hash = value
    elif target == 'email':
        users[id].email = value
    else:
        return False

    # 書き込んで結果を返す
    return WriteUsersCsv(path, users)


def GetUserStats(task_id) -> {}:
    # Task情報からmetricを読み込む
    task = Task(task_id)
    
    # ユーザ情報を読み込む
    users = ReadUsersCsv(USER_CSV_PATH)

    file_paths = glob.glob(os.path.join(Task.TASKS_DIR, task_id, OUTPUT_DIR_NAME, "user", "*.csv"))
    stats = {}
    for file_path in file_paths:
        user_id = os.path.splitext(os.path.basename(file_path))[0]
        if not user_id in users:
            continue
        user_name = users[user_id].name
        stats[user_id] = []
        
        with open(file_path, "r", encoding='utf-8') as csv_file:
            line = csv_file.readline() # ヘッダ読み飛ばし
            while True:
                line = csv_file.readline()
                if not line:
                    break

                stats[user_id].append(Stats(line, user_name, task.metric, task.goal, user_id))

        # ひとつもstatsがなかった場合はキーを削除
        if len(stats[user_id]) == 0:
            stats.pop(user_id)

    return stats


def menuHTML(page, task_id="", url_from="", admin=False):
    html = """
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">IR <span style="color:#00a497">T</span>asks</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarSupportedContent">
                    <ul class="navbar-nav me-auto mb-2 mb-lg-0">
    """

    if page != Page.HOME:
        html += """
                        <li class="nav-item">
                            <a class="nav-link{1} href="/{5}/task">{0}</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link{2} href="/{5}/board">評価結果</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link{3} href="/{5}/log">履歴</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link{4} href="/{5}/upload">提出</a>
                        </li>
        """.format(
            TASK[task_id].name,
            " active\" aria-current=\"page\"" if page == Page.TASK else "\"",
            " active\" aria-current=\"page\"" if page == Page.BOARD else "\"",
            " active\" aria-current=\"page\"" if page == Page.LOG else "\"",
            " active\" aria-current=\"page\"" if page == Page.UPLOAD else "\"",
            task_id
        )

        if admin:
            html += """
                        <li class="nav-item">
                            <a class="nav-link{0} href="/{1}/admin">管理者</a>
                        </li>
            """.format(
                " active\" aria-current=\"page\"" if page == Page.ADMIN else "\"",
                task_id
            )

    html += f"""
                    </ul>
                    <ul class="navbar-nav ml-auto mb-2 mb-lg-0">
                        <li class="nav-item">
                            <a id="login-user-name" class="nav-link active" aria-current="page" href="/user/info{'?from=' + url_from if url_from != '' else ''}"></a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
    """

    return Markup(html)


def EvaluatedValueStyle(metric:Task.Metric, evaluated_value, goal) -> str:
    achieve = False
    if metric == Task.Metric.Accuracy:
        if evaluated_value >= goal:
            achieve = True
    elif metric == Task.Metric.MAE:
        if evaluated_value <= goal:
            achieve = True
    
    return ' style="color:#0dcaf0"' if achieve else ''


def Achieve(task:Task, stats:Stats):
    # コンテスト中の場合は非表示
    if task.type == Task.TaskType.Contest and datetime.datetime.now() < task.end_date:
        return ''

    result = f'<span style="color:#0dcaf0">{"★" if task.type == Task.TaskType.Contest else "☆"}</span>'
    if task.metric == Task.Metric.Accuracy:
        if stats.train < task.goal or stats.valid < task.goal or (task.type == Task.TaskType.Contest and stats.test < task.goal):
            result = ''
    elif task.metric == Task.Metric.MAE:
        if stats.train > task.goal or stats.valid > task.goal or (task.type == Task.TaskType.Contest and stats.test > task.goal):
            result = ''

    return result


def CreateTableRow(stats, task:Task, test=False, message=False, memo=False, visible_invalid_result=False, unlock=False):
    html_user = ""
    html_user += f'<tr>'
    html_user += f'<td>{stats.username} {Achieve(task, stats)}</td>'
    if unlock and AchieveGoal(task, stats):
        html_user += f'<td><a href="/source/{task.id}/{stats.filename}" class="link-info">{stats.datetime}</a></td>'
    else:
        html_user += f'<td>{stats.datetime}</td>'

    html_temp = ""
    if task.metric == Task.Metric.Accuracy:
        if stats.train < 0:
            if visible_invalid_result:
                html_temp += ('<td>-</td><td>-</td><td>-</td>' if test else '<td>-</td><td>-</td>')
            else:
                return ""
        else:
            html_temp += f'<td{EvaluatedValueStyle(task.metric, stats.train, task.goal)}>{stats.train * 100:.2f} %</td>'
            html_temp += f'<td{EvaluatedValueStyle(task.metric, stats.valid, task.goal)}>{stats.valid * 100:.2f} %</td>'
            if test:
                if task.type == Task.TaskType.Contest:
                    if unlock:
                        html_temp += f'<td{EvaluatedValueStyle(task.metric, stats.test, task.goal)}>{stats.test * 100:.2f} %</td>'
                    else:
                        html_temp += f'<td>?</td>'
                else:
                    html_temp += f'<td>-</td>'
    
    elif task.metric == Task.Metric.MAE:
        if stats.train < 0:
            if visible_invalid_result:
                html_temp += ('<td>-</td><td>-</td><td>-</td>' if test else '<td>-</td><td>-</td>')
            else:
                return ""
        else:
            try:
                html_temp += f'<td{EvaluatedValueStyle(task.metric, stats.train, task.goal)}>{stats.train:.3f}</td>'
                html_temp += f'<td{EvaluatedValueStyle(task.metric, stats.valid, task.goal)}>{stats.valid:.3f}</td>'
                if test:
                    if task.type == Task.TaskType.Contest:
                        if unlock:
                            html_temp += f'<td{EvaluatedValueStyle(task.metric, stats.test, task.goal)}>{stats.test:.3f}</td>'
                        else:
                            html_temp += f'<td>?</td>'
                    else:
                        html_temp += f'<td>-</td>'
            except:
                html_temp += ('<td>-</td><td>-</td><td>-</td>' if test else '<td>-</td><td>-</td>')

    html_user += html_temp

    if memo:
        html_user += f'<td>{stats.memo}</td>'
    if message:
        html_user += f'<td>{stats.message}</td>'
    html_user += f'</tr>'

    return html_user


def CreateBoardTable(stats_list, task:Task, test=False, message=False, memo=False, unlock=False, table_id='sortable-table'):
    def metricName(metric: Task.Metric):
        if metric == Task.Metric.Accuracy:
            return '正解率'
        elif metric == Task.Metric.MAE:
            return '平均絶対誤差'
        else:
            return ''

    num_col = 0
    html_table = ""
    html_table += f"<table class=\"table table-dark\" id=\"{table_id}\">"
    html_table += "<thead><tr>"
    html_table += f"<th id=\"th-{num_col}\">参加者</th>"
    num_col += 1
    html_table += f"<th id=\"th-{num_col}\">提出日時</th>"
    num_col += 1
    html_table += f"<th id=\"th-{num_col}\">train(配布){metricName(task.metric)}</th>"
    num_col += 1
    html_table += f"<th id=\"th-{num_col}\">valid{metricName(task.metric)}</th>"
    num_col += 1
    if test:
        html_table += f"<th id=\"th-{num_col}\">test{metricName(task.metric)}</th>"
        num_col += 1
    if memo:
        html_table += f"<th id=\"th-{num_col}\">メモ</th>"
        num_col += 1
    if message:
        html_table += f"<th id=\"th-{num_col}\">メッセージ</th>"
        num_col += 1
    html_table += "</tr></thead>"
    html_table += "<tbody>"

    for stats in stats_list:
        html_table += CreateTableRow(stats, task, test=test, message=message, memo=memo, unlock=unlock)

    html_table += "</tbody>"
    html_table += "</table>"

    return html_table, num_col


def CreateInProcHtml(task_id):
    # ユーザ情報を読み込む
    users = ReadUsersCsv(USER_CSV_PATH)

    inproc_text = ''
    for user_id in users:
        if os.path.exists(os.path.join(Task.TASKS_DIR, task_id, OUTPUT_DIR_NAME, "user", f"{user_id}_inproc")):
            inproc_text += f"{users[user_id].name} さんの評価を実行中です。<br>"

    return inproc_text + '<br>'


def CreateMyTaskTable(user_id) -> str:
    submits = []

    # user_idのstatsをTaskごとに取得
    for task_id, task in TASK.items():
        task:Task = task
        user_stats = GetUserStats(task_id)
        if user_id in user_stats:
            stats = user_stats[user_id]

            # 表示最優先の成績を選択
            best_stats = Stats.GetBestStats(stats, task)
            if best_stats is not None:
                submit:Submit = Submit()
                submit.stats = best_stats
                submit.task = task
                submits.append(submit)

    # 提出日時でソート
    sorted_submits = sorted(submits, key=lambda x: x.stats.datetime, reverse=True)
    
    # 表のHTMLを作成
    html_table = '<table class="table table-dark">'
    html_table += "<thead><tr>"
    html_table += "<th>提出日時</th>"
    html_table += "<th>Task</th>"
    html_table += "<th>Goal</th>"
    html_table += "<th>train</th>"
    html_table += "<th>valid</th>"
    html_table += "<th>test</th>"
    html_table += "</tr></thead>"
    html_table += "<tbody>"

    for submit in sorted_submits:
        html_table += CreateSubmitTableRow(submit, goal=True, test=True, memo=False, message=False)

    html_table += "</tbody>"
    html_table += "</table>"

    return html_table


class Submit:
    stats: Stats
    task: Task


def CreateSubmitTableRow(submit:Submit, visible_invalid_data:bool=False, goal=False, test=False, memo:bool=True, message:bool=True):
    try:
        html_submit = ""
        html_submit += f'<tr>'
        html_submit += f'<td><a href="/source/{submit.task.id}/{submit.stats.filename}" class="link-info">{submit.stats.datetime}</a></td>'
        html_submit += f'<td><a href="/{submit.task.id}/task" class="link-info">{submit.task.name}</a></td>'

        html_temp = ""
        if submit.task.metric == Task.Metric.Accuracy:
            if goal:
                html_temp += f'<td>正解率 <span style="color:#0dcaf0">{submit.task.goal*100:.0f}</span> &percnt; 以上 {Achieve(submit.task, submit.stats)}</td>'
            if submit.stats.train < 0:
                if visible_invalid_data:
                    html_temp += '<td>-</td><td>-</td>' if not test else '<td>-</td><td>-</td><td>-</td>'
                else:
                    return ""
            else:
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.train, submit.task.goal)}>{submit.stats.train * 100:.2f} &percnt;</td>'
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.valid, submit.task.goal)}>{submit.stats.valid * 100:.2f} &percnt;</td>'
                if test:
                    if submit.task.type == Task.TaskType.Quest:
                        html_temp += '<td>-</td>'
                    elif submit.task.type == Task.TaskType.Contest:
                        unlock = False
                        if AchieveGoal(submit.task, submit.stats):
                            # Questであればいつでも、Contestであれば期間終了後にロック解除
                            if datetime.datetime.now() >= submit.task.end_date: 
                                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.test, submit.task.goal)}>{submit.stats.test * 100:.2f} &percnt;</td>'
                                unlock = True
                        if not unlock:
                            html_temp += f'<td>?</td>'

        elif submit.task.metric == Task.Metric.MAE:
            if goal:
                html_temp += f'<td>MAE <span style="color:#0dcaf0">{submit.task.goal:.1f}</span> 以下 {Achieve(submit.task, submit.stats)}</td>'
            try:
                if submit.stats.train < 0:
                    if visible_invalid_data:
                        html_temp += '<td>-</td><td>-</td>' if not test else '<td>-</td><td>-</td><td>-</td>'
                    else:
                        return ""
                else:
                    html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.train, submit.task.goal)}>{submit.stats.train:.3f}(MAE)</td>'
                    html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.valid, submit.task.goal)}>{submit.stats.valid:.3f}(MAE)</td>'
                    if test:
                        if submit.task.type == Task.TaskType.Quest:
                            html_temp += '<td>-</td>'
                        elif submit.task.type == Task.TaskType.Contest:
                            unlock = False
                            if AchieveGoal(submit.task, submit.stats):
                                # Questであればいつでも、Contestであれば期間終了後にロック解除
                                if datetime.datetime.now() >= submit.task.end_date: 
                                    html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.test, submit.task.goal)}>{submit.stats.test:.3f}(MAE)</td>'
                                    unlock = True
                            if not unlock:
                                html_temp += f'<td>?</td>'
            except:
                html_temp += '<td>-</td><td>-</td>'

        html_submit += html_temp

        if memo:
            html_submit += f'<td>{submit.stats.memo}</td>'
        if message:
            html_submit += f'<td>{submit.stats.message}</td>'

        html_submit += f'</tr>'
    except Exception as e:
        print(e)
        return ""

    return html_submit


def CreateSubmitTable(user_id) -> str:
    submits = []

    # user_idのstatsをTaskごとに取得
    for task_id, task in TASK.items():
        stats_temp = GetUserStats(task_id)
        if user_id in stats_temp:
            for item in stats_temp[user_id]:
                submit: Submit = Submit()
                submit.stats = item
                submit.task = task
                submits.append(submit)

    # 提出日時でソート
    sorted_submits = sorted(submits, key=lambda x: x.stats.datetime, reverse=True)
    
    # 表のHTMLを作成
    html_table = '<table class="table table-dark">'
    html_table += "<thead><tr>"
    html_table += "<th>提出日時</th>"
    html_table += "<th>Task</th>"
    html_table += "<th>train</th>"
    html_table += "<th>valid</th>"
    html_table += "<th>メモ</th>"
    html_table += "<th>メッセージ</th>"
    html_table += "</tr></thead>"
    html_table += "<tbody>"

    for submit in sorted_submits:
        html_table += CreateSubmitTableRow(submit, True)

    html_table += "</tbody>"
    html_table += "</table>"

    return html_table


def CreateUserTable() -> str:
    # 表のHTMLを作成
    html_table = '<table class="table table-dark">'
    html_table += "<thead><tr>"
    html_table += "<th>ID</th>"
    html_table += "<th>Name</th>"
    html_table += "<th>Email</th>"
    html_table += "<th>Num Submit</th>"
    html_table += "<th>Latest Submit</th>"
    html_table += "<th>Task</th>"
    html_table += "</tr></thead>"
    html_table += "<tbody>"

    # ユーザ情報を読み込む
    users = ReadUsersCsv(USER_CSV_PATH)

    for user_id, user in users.items():
        user_data:UserData = user

        # 最新の提出を抽出
        latest_datetime:datetime.datetime = datetime.datetime(1984, 4, 22)
        latest_submit_task:Task = None
        num_submit = 0
        for task_id, task in TASK.items():
            stats_temp = GetUserStats(task_id)
            if not user_id in stats_temp:
                continue
            num_submit += len(stats_temp[user_id])
            for item in stats_temp[user_id]:
                stats:Stats = item
                if stats.datetime > latest_datetime:
                    latest_datetime = stats.datetime
                    latest_submit_task = task

        html_table += "<tr>"
        html_table += f"<td>{user_data.id}</td>"
        html_table += f"<td>{user_data.name}</td>"
        html_table += f"<td>{user_data.email}</td>"
        html_table += f"<td>{num_submit}</td>"
        html_table += f"<td>{latest_datetime if latest_submit_task is not None else '-'}</td>"
        if latest_submit_task is not None:
            html_table += f'<td><a href="/{latest_submit_task.id}/task" class="link-info">{latest_submit_task.name}</a></td>'
        else:
            html_table += '<td>-</td>'
        html_table += "</tr>"

    html_table += "</tbody>"
    html_table += "</table>"

    return html_table


def CreateTaskTable(tasks) -> str:
    # 表のHTMLを作成
    html_table = '<table class="table table-dark">'
    html_table += "<thead><tr>"
    html_table += "<th>ID</th>"
    html_table += "<th>Name</th>"
    html_table += "<th>開始日</th>"
    html_table += "<th>終了日</th>"
    html_table += "<th>Type</th>"
    html_table += "<th>Metric</th>"
    html_table += "<th>Goal</th>"
    html_table += "<th>制限時間[s/data]</th>"
    html_table += "<th>停止</th>"
    html_table += "<th>変更</th>"
    html_table += "</tr></thead>"
    html_table += "<tbody>"

    for task_id, item in tasks.items():
        task:Task = item
        html_table += f'<form method="POST">'
        html_table += f'<input type="hidden" name="task-id" value="{task_id}">'
        html_table += f"<tr>"
        html_table += f"<td>{task.id}</td>"
        html_table += f"<td>{task.name}</td>"
        html_table += f'<td><input type="date" name="start-date" value="{task.start_date.date()}" class="bg-dark text-white"></td>'
        html_table += f'<td><input type="date" name="end-date" value="{task.end_date.date()}" class="bg-dark text-white"></td>'
        html_table += f"<td>{task.type.name}</td>"
        html_table += f"<td>{task.metric.name}</td>"
        html_table += f'<td><input type="number" name="goal" value="{task.goal}" step="0.1" class="bg-dark text-white"></td>'
        html_table += f'<td><input type="number" name="timelimit-per-data" value="{task.timelimit_per_data}" step="0.1" class="bg-dark text-white"></td>'
        html_table += f'<td>&nbsp;&nbsp;&nbsp;<input type="checkbox" name="suspend" class="form-check-input"{" checked" if task.suspend else ""}>&nbsp;&nbsp;&nbsp;</td>'
        html_table += f'<td><input type="submit" value="変更" class="btn btn-outline-info"></td>'
        html_table += f"</tr>"
        html_table += f'</form>'

    html_table += "</tbody>"
    html_table += "</table>"

    return html_table


def VerifyEmailAndPassword(email, password):
    # ユーザ情報を読み込む
    users = ReadUsersCsv(USER_CSV_PATH)
    
    # 認証を行う
    verified = False
    user_data = None
    for id, user_data in users.items():
        if user_data.email == email:
            pass_hash = user_data.pass_hash
            if check_password_hash(pass_hash, password):
                verified = True
            break
    
    return verified, user_data


def VerifyIdAndKey(user_id, user_key):
    # ユーザ情報を読み込む
    users = ReadUsersCsv(USER_CSV_PATH)
    
    # 認証を行う
    verified = False
    if user_id in users:
        user_data = users[user_id]
        if user_data.key == user_key:
            verified = True
    else:
        user_data = UserData()

    return verified, user_data


def VerifyByCookie(request):
    verified = False
    user_data = None
    admin = False

    try:
        user_info = request.cookies.get(COOKIE_KEY)
        if user_info is not None:
            user_info = json.loads(user_info)
            user_id = user_info['id']
            user_key = user_info['key']
            verified, user_data = VerifyIdAndKey(user_id, user_key)
            if not verified:
                raise(ValueError())
            admin = True if SETTING["admin"]["email"] == user_data.email else False                
    except:
        print("cookieによる認証に失敗しました。")

    return verified, user_data, admin


def AchieveGoal(task:Task, stats:Stats):
    results = [stats.train, stats.valid]

    if task.type == Task.TaskType.Contest:
        results.append(stats.test)

    for result in results:
        if task.metric == Task.Metric.Accuracy:
            if result < task.goal:
                return False
        elif task.metric == Task.Metric.MAE:
            if result > task.goal:
                return False

    return True


@app.route('/')
def index():
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)

    today = datetime.datetime.now()
    task_list_quest = []
    task_list_open = []
    task_list_closed = []
    task_list_prepare = []
    for key, value in TASK.items():
        task:Task = value
        if task.suspend:
            continue

        task_period = ''
        if task.start_date <= today:
            if task.type == Task.TaskType.Contest:
                if task.end_date <= today:
                    # 終了後
                    task_period = f'開催期間後' 
                else:
                    # 開催中
                    task_period = f'{task.start_date.strftime("%Y-%m-%d")}～{(task.end_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")}' 
            elif task.type == Task.TaskType.Quest:
                task_period = f'開催中' 
        else:
            # スタート前
            task_period = f'開始前({task.start_date.strftime("%Y-%m-%d")}～{(task.end_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")})'

        info = {
            'id': key,
            'name': task.name,
            'explanation': task.explanation,
            'period': task_period
        }

        if value.start_date <= today:
            # スタート後
            if task.type == Task.TaskType.Quest:
                # Questはいつまでも開かれている
                task_list_quest.append(info)
            elif task.type == Task.TaskType.Contest:
                # Contestは開催期間により振り分け
                if value.end_date > today:
                    task_list_open.append(info)
                else:
                    task_list_closed.append(info)
        else:
            # スタート前
            if admin:
                task_list_prepare.append(info)
    
    return render_template('index.html', task_list_open=task_list_open, task_list_closed=task_list_closed, task_list_quest=task_list_quest, task_list_prepare=task_list_prepare, name_contest=SETTING["name"]["contest"], menu=menuHTML(Page.HOME, url_from="/"), admin=admin)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/img'), 'favicon.ico', )


@app.route('/join', methods=['GET', 'POST'])
def join():
    from_url = request.args.get('from')
    if from_url is None:
        from_url = "/"

    if request.method == 'GET':
        return render_template(f'join.html', from_url=from_url, message="")
    
    elif request.method == 'POST':
        try:
            email = request.form['inputEmail']
            password = request.form['inputPassword']
            password_verify = request.form['inputPasswordVerify']
            next_url = request.form['nextUrl']
        except:
            return render_template(f'join.html', from_url=from_url, message="入力データを受け取れませんでした。")
        
        # 2つのパスワード入力の一致チェック
        if password != password_verify:
            return render_template(f'join.html', from_url=from_url, message="再入力したパスワードが一致していません。")
        
        # ユーザ情報を読み込む
        users = ReadUsersCsv(USER_CSV_PATH)

        # email重複チェック
        duplicate = False
        for user_id, user_data in users.items():
            if email == user_data.email:
                duplicate = True
                break
        if duplicate:
            Log.write(f"Failed to create account. the email already exist. email: {email}")
            return render_template(f'join.html', from_url=from_url, message="そのEmail addressは既に登録されています。")
        
        # ID発行
        user_id = ""
        for i in range(10000):
            id = str(uuid.uuid4()).split('-')[0]
            if id in users:
                continue
            else:
                user_id = id
                break

        if user_id == "":
            Log.write(f"Failed to create ID. email: {email}")
            return render_template(f'join.html', from_url=from_url, message="IDを発行できませんでした。")
        
        # パスワードをハッシュ化
        pass_hash = generate_password_hash(password, salt_length=21)
        
        # 本人確認用のキーを作成
        user_key = str(uuid.uuid4()).split('-')[0]

        # ユーザ登録
        name = email.split('@')[0]
        success = AddUsersCsv(USER_CSV_PATH, user_id, email, name, pass_hash, user_key)

        if not success:
            Log.write(f"Failed to create account. AddUsersCsv() error. email: {email}")
            return render_template(f'join.html', from_url=from_url, message="ユーザ情報を登録できませんでした。")

        Log.write(f"Success to create account. email: {email}")

        return render_template(f'user.html', from_url=from_url, user_email=email, user_id=user_id, user_key=user_key, user_name=name, next_url=next_url, login="true", update_user_data="true")


@app.route('/login', methods=['GET', 'POST'])
def login():
    from_url = request.args.get('from')
    if from_url is None:
        from_url = "/"

    if request.method == 'GET':
        return render_template(f'login.html', from_url=from_url, message="", email_admin=SETTING["admin"]["email"])
    
    elif request.method == 'POST':
        try:
            email = request.form['inputEmail']
            password = request.form['inputPassword']
            next_url = request.form['nextUrl']
        except:
            return render_template(f'login.html', from_url=from_url, message="入力データを受け取れませんでした。", email_admin=SETTING["admin"]["email"])
        
        # emailとパスワードで照合
        verified, user_data = VerifyEmailAndPassword(email, password)
        if not verified:
            Log.write(f"Failed to log in. email: {email}")
            return render_template(f'login.html', from_url=from_url, message="Email addressかPasswordが誤っています。", email_admin=SETTING["admin"]["email"])

        Log.write(f"Success to log in. email: {email}")

        # 認証OKなので本人確認用のキーを作成して渡す
        user_key = str(uuid.uuid4()).split('-')[0]

        # キーを保存
        UpdateUsersCsv(USER_CSV_PATH, user_data.id, 'key', user_key)

        # ユーザ情報をクッキーに書き込み
        response = make_response(
            render_template(f'user.html', from_url=from_url, user_email=email, user_id=user_data.id, user_key=user_key, user_name=user_data.name, next_url=next_url, login="true", update_user_data="true")
        )
        user_info = {'id':user_data.id, 'key':user_key}
        expires = int(datetime.datetime.now().timestamp()) + COOKIE_AGE_SEC
        response.set_cookie(COOKIE_KEY, value=json.dumps(user_info), expires=expires)

        return response


@app.route('/logout')
def logout():
    # ユーザ認証(認証できなくてもログアウト処理は実施)
    verified, user_data, admin = VerifyByCookie(request)

    # ユーザ情報をクッキーに書き込み
    response = make_response(
        render_template(f'logout.html')
    )
    user_info = {'id':'', 'key':''}
    expires = int(datetime.datetime.now().timestamp()) + COOKIE_AGE_SEC
    response.set_cookie(COOKIE_KEY, value=json.dumps(user_info), expires=expires)

    Log.write(f"Success to log out. email: {user_data.email if verified else 'not verified'}")

    return response


@app.route('/user/info', methods=['GET', 'POST'])
def user():
    if request.method == 'GET':
        from_url = request.args.get('from')
        return render_template(f'user.html', login="false", from_url=from_url if from_url is not None else "/")

    elif request.method == 'POST':
        try:
            user_id = request.form['userID']
            user_key = request.form['userKey']
            verified, user_data = VerifyIdAndKey(user_id, user_key)
            if not verified:
                raise(ValueError())
        except:
            return render_template(f'user.html', message='ユーザ認証に失敗しました。')
        
        new_name = ''
        message = ''
        try:
            if 'buttonChangeName' in request.form:
                # 名前を変更
                new_name = request.form['newName']
                success = UpdateUsersCsv(USER_CSV_PATH, user_id, 'name', new_name)
                if not success:
                    message = 'ユーザ情報の更新に失敗しました。'
                    raise(ValueError())
                message = 'ユーザ名を変更しました。'
                Log.write(f"Success to change user name. user_id: {user_id}")
            elif 'buttonChangePassword' in request.form:
                # パスワードを変更
                password = request.form['inputPassword']
                password_verify = request.form['inputPasswordVerify']

                # 2つのパスワード入力の一致チェック
                if password != password_verify:
                    message = '再入力したパスワードが一致していません。'
                    raise(ValueError())
                
                # パスワードをハッシュ化
                pass_hash = generate_password_hash(password, salt_length=21)
                success = UpdateUsersCsv(USER_CSV_PATH, user_id, 'pass_hash', pass_hash)
                if not success:
                    message = 'ユーザ情報の更新に失敗しました。'
                    raise(ValueError())
                message = 'パスワードを変更しました。'
                Log.write(f"Success to change password. user_id: {user_id}")
        except:
            return render_template(f'user.html', message=message)

        return render_template(f'user.html', user_name=new_name, update_user_data="true", message=message)


@app.route('/my-task-table/<user_id>/<user_key>')
def my_task_table(user_id, user_key):
    # 認証
    verified, user_data = VerifyIdAndKey(user_id, user_key)
    if not verified:
        return redirect(url_for('login'))
    
    # 提出Taskテーブルを作成
    table_html = CreateMyTaskTable(user_id)

    return table_html


@app.route('/submit-table/<user_id>/<user_key>')
def submit_table(user_id, user_key):
    # 認証
    verified, user_data = VerifyIdAndKey(user_id, user_key)
    if not verified:
        return redirect(url_for('login'))
    
    # 提出テーブルを作成
    table_html = CreateSubmitTable(user_id)

    return table_html


@app.route('/source/<task_id>/<filename>')
def source(task_id, filename):
    file_path = os.path.join(Task.TASKS_DIR, task_id, USER_MODULE_DIR_NAME, filename)
    if not os.path.exists(file_path):
        return render_template(f'source.html', filename='ファイルが見つかりません')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 元のファイル名を復元
        filename_split = filename.split('_')
        filename_head = f"{filename_split[0]}_{filename_split[1]}_{filename_split[2]}_{filename_split[3]}_"
        filename_org = filename.replace(filename_head, '')
    except Exception as e:
        return render_template(f'source.html', filename='ファイルを読み込めません')

    return render_template(f'source.html', source=content, filename=filename_org)


@app.route('/verify/<user_id>/<user_key>', methods=['GET'])
def verify(user_id, user_key):
    verified = False
    
    try:
        verified, user_data = VerifyIdAndKey(user_id, user_key)
    except:
        verified = False

    return "true" if verified else "false"


@app.route('/<task_id>/')
def task_index(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))

    return redirect(url_for('task', task_id=task_id))


@app.route('/<task_id>/timestamp', methods=['GET'])
def get_timestamp(task_id):
    try:
        with open(os.path.join(Task.TASKS_DIR, task_id, OUTPUT_DIR_NAME, "timestamp.txt"), "r", encoding='utf-8') as f:
            timestamp = f.read()
    except:
        timestamp = ''

    return timestamp


@app.route('/<task_id>/card.png')
def get_taskcard(task_id):
    return send_from_directory(os.path.join(Task.TASKS_DIR, task_id), "card.png")


@app.route("/<task_id>/task")
def task(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))

    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)

    # タスク情報を読み込む
    task:Task = Task(task_id)

    return render_template(f'tasks/{task_id}/index.html',
                           menu=menuHTML(Page.TASK, task_id, url_from=f"/{task_id}/task", admin=admin),
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           goal=Task.GoalText(task.metric, task.goal))


@app.route("/<task_id>/board", methods=['GET'])
def board(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))

    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    
    # タスク情報を読み込む
    task:Task = Task(task_id)

    # ユーザ成績を読み込む
    user_stats = GetUserStats(task_id)

    # ユーザごとに表示最優先の成績を選択
    best_stats_every_user = []
    best_stats_in_contest = []
    my_stats = None
    for user_name, stats in user_stats.items():
        best_stats:Stats = Stats.GetBestStats(stats, task)
        if best_stats is not None:
            best_stats_every_user.append(best_stats)
            if verified and best_stats.userid == user_data.id:
                my_stats = best_stats

        # コンテスト期間中の各ユーザベスト
        if task.type == Task.TaskType.Contest and datetime.datetime.now() >= task.end_date:
            stats_in_contest = []
            for item in stats:
                one_stats:Stats = item
                if one_stats.datetime >= task.start_date and one_stats.datetime < task.end_date:
                    stats_in_contest.append(one_stats)
            best_stats:Stats = Stats.GetBestStats(stats_in_contest, task)
            if best_stats is not None:
                best_stats_in_contest.append(best_stats)

    # 日付順にソート
    sorted_stats_list = sorted(best_stats_every_user, key=lambda x: x.datetime, reverse=True)
    
    # test成績順にソート
    if len(best_stats_in_contest) > 0:
        sorted_stats_list_in_contest = sorted(best_stats_in_contest, key=lambda x: x.test, reverse=True)
 
    # unlock判定
    unlock = False
    if verified and my_stats is not None:
        # ユーザの認証ができている場合、このタスクの目標を達成しているか確認
        if AchieveGoal(task, my_stats):
            # Questであればいつでも、Contestであれば期間終了後にロック解除
            if task.type == Task.TaskType.Quest:
                unlock = True
            elif task.type == Task.TaskType.Contest and datetime.datetime.now() >= task.end_date: 
                unlock = True

    # 表を作成
    html_table, num_col = CreateBoardTable(sorted_stats_list, task, unlock=unlock, test=True if task.type == Task.TaskType.Contest else False)

    # コンテスト終了時の成績表を作成
    html_contest_result = None
    if len(best_stats_in_contest) > 0:
        html_contest_result, num_col = CreateBoardTable(sorted_stats_list_in_contest, task, unlock=unlock, test=True)

    return render_template('board.html',
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           table_board=Markup(html_table),
                           table_contest_result=Markup(html_contest_result) if html_contest_result is not None else None,
                           menu=menuHTML(Page.BOARD, task_id, url_from=f"/{task_id}/board", admin=admin),
                           inproc_text=Markup(CreateInProcHtml(task_id)),
                           goal=Task.GoalText(task.metric, task.goal),
                           num_col=num_col, task_id=task_id
                           )


@app.route("/<task_id>/log")
def log(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))
    
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)

    # タスク情報を読み込む
    task:Task = Task(task_id)

    # ユーザ成績を読み込む
    user_stats = GetUserStats(task_id)

    # 辞書をリスト化してソート
    unlock = False
    stats_list = []
    for stats in user_stats.values():
        for item in stats:
            stats_list.append(item)
            if verified and item.userid == user_data.id and AchieveGoal(task, item):
                # Questであればいつでも、Contestであれば期間終了後にロック解除
                if task.type == Task.TaskType.Quest:
                    unlock = True
                elif task.type == Task.TaskType.Contest and datetime.datetime.now() >= task.end_date: 
                    unlock = True

    sorted_stats_list = sorted(stats_list, key=lambda x: x.datetime, reverse=True)

    # 表を作成
    html_table, num_col = CreateBoardTable(sorted_stats_list, task, unlock=unlock, test=True if task.type == Task.TaskType.Contest else False)

    return render_template('log.html',
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           table_log=Markup(html_table),
                           menu=menuHTML(Page.LOG, task_id, url_from=f"/{task_id}/log", admin=admin), 
                           inproc_text=Markup(CreateInProcHtml(task_id)), 
                           num_col=num_col,
                           task_id=task_id,
                           goal=Task.GoalText(task.metric, task.goal))


@app.route('/<task_id>/upload', methods=['GET', 'POST'])
def upload_file(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))
    
    task:Task = TASK[task_id]
    
    if task.suspend:
        return redirect(url_for('index'))
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    msg = ""

    if request.method == 'POST':
        file = request.files['file']
        if not file:
            msg = "ファイルが選択されていません。"
        elif not allowed_file(file.filename):
            msg = "アップロードできるファイルは.pyのみです。"
        else:
            user_id = request.form['user_id']
            user_key = request.form['user_key']
            
            verified, user_data = VerifyIdAndKey(user_id, user_key)
            if not verified:
                msg = "ユーザ認証に失敗しました。"
            else:
                try:
                    save_dir = os.path.join(Task.TASKS_DIR, task_id, UPLOAD_DIR_NAME, user_id)

                    # まだディレクトリが存在しなければ作成(Taskのディレクトリがなければそれも生成)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    if os.path.exists(save_dir):
                        new_filename = secure_filename(file.filename)
                        file.save(os.path.join(save_dir, new_filename))
                        msg = f'{file.filename}がアップロードされました。'

                        # メモを保存
                        if request.form['memo'] != "":
                            with open(os.path.join(save_dir, new_filename + '.txt'), mode='w', encoding='utf-8') as f:
                                f.write(request.form['memo'])
                    else:
                        raise(ValueError("アップロード先のディレクトリが存在しません。"))
                except:
                    msg = "アップロードに失敗しました。"

    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)

    return render_template('upload.html', task_id=task_id, task_name=task.name, message=msg, menu=menuHTML(Page.UPLOAD, task_id, url_from=f"/{task_id}/upload", admin=admin), url_from=f"/{task_id}/upload", time_limit=task.timelimit_per_data)
  

@app.route('/<task_id>/admin')
def admin(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))
    
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    if not admin:
        return redirect(url_for('index'))
    
    # タスク情報を読み込む
    task:Task = Task(task_id)

    # ユーザ成績を読み込む
    user_stats = GetUserStats(task_id)

    # 辞書をリスト化してソート
    stats_list = []
    for stats in user_stats.values():
        for item in stats:
            stats_list.append(item)
    sorted_stats_list = sorted(stats_list, key=lambda x: x.datetime, reverse=True)

    # 表を作成
    html_table, num_col = CreateBoardTable(sorted_stats_list, task, test=True, message=True, unlock=True)

    return render_template('log.html',
                           task_id=task_id,
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           table_log=Markup(html_table),
                           menu=menuHTML(Page.ADMIN, task_id, url_from=f"/{task_id}/admin", admin=admin),
                           inproc_text=Markup(CreateInProcHtml(task_id)),
                           num_col=num_col)

@app.route('/admin', methods=['GET', 'POST'])
def manage():
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    if not admin:
        return redirect(url_for('index'))
    
    # タスク一覧の再読み込み
    global TASK
    TASK = Task.readTasks()

    # タスク情報の変更
    if request.method == 'POST':
        try:
            target_task_id = request.form["task-id"]
            if not target_task_id in TASK:
                raise(ValueError())
            
            # タスク情報の書き換え
            if target_task_id is not None:
                task:Task = TASK[target_task_id]
                task.start_date = datetime.datetime.strptime(request.form["start-date"], '%Y-%m-%d')
                task.end_date = datetime.datetime.strptime(request.form["end-date"], '%Y-%m-%d')
                task.goal = float(request.form["goal"])
                task.timelimit_per_data = float(request.form["timelimit-per-data"])
                task.suspend = True if "suspend" in request.form else False
                
            # ファイル出力
            success = task.save()
            Log.write(f"{Task.FILENAME_TASK_JSON} of {target_task_id} {'was uplooaded.' if success else 'cannot be uploaded.'}")

            # タスク一覧の再読み込み
            TASK = Task.readTasks()

        except:
            print("Task情報の書き換えに失敗")
   
    return render_template('admin.html',
                           user_table=Markup(CreateUserTable()),
                           task_table=Markup(CreateTaskTable(TASK)),
                           log_table=Markup(Log.createTable()))


if __name__ == "__main__":
    try:
        json_open = open(SETTING_JSON_PATH, 'r', encoding='utf-8')
        SETTING = json.load(json_open)          
    except:
        print(f"settingファイルを開けません: {SETTING_JSON_PATH}")
        exit()

    # タスク一覧
    TASK = Task.readTasks()

    # アプリ開始
    app.run(debug=False, host='0.0.0.0', port=5000)
