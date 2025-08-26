from flask import Flask, render_template, request, redirect, url_for, make_response, send_from_directory, send_file
from markupsafe import Markup
from flask_httpauth import HTTPBasicAuth, HTTPDigestAuth
from werkzeug.utils import secure_filename
import json
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import math

import os
import glob
import datetime
from enum import Enum
import shutil
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from module.task import Task, Stats, Log
from module.user import User

from tasks_system import FILENAME_DATASET_JSON


ALLOWED_EXTENSIONS = set(['py'])
TASK = {}
SETTING = None
SETTING_JSON_PATH = r"./data/setting.json"
COOKIE_KEY = "user"
COOKIE_AGE_SEC = 60 * 60 * 24 * 365


class Page(Enum):
    HOME = 0,
    TASK = 1,
    BOARD = 2,
    LOG = 3,
    UPLOAD = 4,
    ADMIN = 9


# Flaskオブジェクトの生成
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024 #ファイルサイズ制限 256MB
app.config['SECRET_KEY'] = 'secret key here'
app.config['TEMPLATES_AUTO_RELOAD'] = True # テンプレートの自動リロードを有効にする設定


def menuHTML(page, task_id="", url_from="", admin=False, user_name=""):
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
                            <a id="login-user-name" class="nav-link active" aria-current="page" href="/user/info{'?from=' + url_from if url_from != '' else ''}">
                                {user_name + ' さんのユーザページ' if user_name != '' else 'ログイン'}
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
    """

    return Markup(html)


def EvaluatedValueStyle(metric:Task.Metric, evaluated_value, goal) -> str:
    achieve = Task.achieveValue(metric, evaluated_value, goal)
    return ' style="color:#0dcaf0"' if achieve else ''


class Submit:
    stats:Stats
    task:Task

    def __init__(self, stats:Stats, task:Task) -> None:
        self.stats = stats
        self.task = task


class UnlockMode(Enum):
    UnlockAll = 0,
    LockAll = 1,
    UnlockAchieveStats = 2


def CreateRecordTable(submit_list, table_id='', visible_invalid_data:bool=False, user_name=False, task_name=False, goal=False, test=False, memo:bool=False, message:bool=False, unlock_mode:UnlockMode=UnlockMode.LockAll):
    # 表の見出しを作成
    num_col = 0
    html_table = ""
    html_table += f"<table class=\"table table-dark\" id=\"{table_id}\">"
    html_table += "<thead><tr>"
    if user_name:
        html_table += f"<th id=\"th-{num_col}\">参加者</th>"
        num_col += 1
    html_table += f"<th id=\"th-{num_col}\">提出日時</th>"
    num_col += 1
    if task_name:
        html_table += f"<th id=\"th-{num_col}\">Task</th>"
        num_col += 1
    if goal:
        html_table += f"<th id=\"th-{num_col}\">Goal</th>"
        num_col += 1
    html_table += f"<th id=\"th-{num_col}\">train</th>"
    num_col += 1
    html_table += f"<th id=\"th-{num_col}\">valid</th>"
    num_col += 1
    if test:
        html_table += f"<th id=\"th-{num_col}\">test</th>"
        num_col += 1
    if memo:
        html_table += f"<th id=\"th-{num_col}\">メモ</th>"
        num_col += 1
    if message:
        html_table += f"<th id=\"th-{num_col}\">メッセージ</th>"
        num_col += 1
    html_table += "</tr></thead>"
    html_table += "<tbody>"

    # 各提出の行を作成
    for submit in submit_list:
        html_table += CreateRecordTableRow(
            submit, visible_invalid_data, user_name, task_name, goal, test, memo, message,
            True if (unlock_mode == UnlockMode.UnlockAll) or (unlock_mode == UnlockMode.UnlockAchieveStats and submit.task.achieve(submit.stats)) else False
        )

    html_table += "</tbody>"
    html_table += "</table>"

    return html_table, num_col


def CreateRecordTableRow(submit:Submit, visible_invalid_data:bool=False, user_name=False, task_name=False, goal=False, test=False, memo:bool=True, message:bool=True, unlock=False):
    try:
        html_submit = ""
        html_submit += f'<tr>'

        # ユーザ名:選択
        if user_name:
            html_submit += f'<td>{User.userIDtoUserName(submit.stats.userid)} {submit.task.achieveStarHTML(submit.stats)}</td>'

        # 提出日時:必ず入る
        if unlock:
            html_submit += f'<td><a href="/source/{submit.task.id}/{submit.stats.filename}" class="link-info">{submit.stats.datetime}</a></td>'
        else:
            html_submit += f'<td>{submit.stats.datetime}</td>'

        # Task名:選択
        if task_name:
            html_submit += f'<td><a href="/{submit.task.id}/task" class="link-info">{submit.task.name}</a></td>'

        html_temp = ""
        if submit.task.metric == Task.Metric.Accuracy:
            # Goal:選択
            if goal:
                html_temp += f'<td>正解率 <span style="color:#0dcaf0">{submit.task.goal*100:.0f}</span> &percnt; 以上 {submit.task.achieveStarHTML(submit.stats)}</td>'
            
            if submit.stats.train < 0:
                if visible_invalid_data:
                    html_temp += '<td>-</td><td>-</td>' if not test else '<td>-</td><td>-</td><td>-</td>'
                else:
                    return ""
            else:
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.train, submit.task.goal)}>{submit.stats.train * 100:.2f} &percnt;</td>'
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.valid, submit.task.goal)}>{submit.stats.valid * 100:.2f} &percnt;</td>'
                
                # Test:選択
                if test:
                    if submit.task.type == Task.TaskType.Quest:
                        html_temp += '<td>-</td>'
                    elif submit.task.type == Task.TaskType.Contest:
                        unlock = False
                        if submit.task.achieve(submit.stats):
                            # Questであればいつでも、Contestであれば期間終了後にロック解除
                            if submit.task.afterContest(): 
                                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.test, submit.task.goal)}>{submit.stats.test * 100:.2f} &percnt;</td>'
                                unlock = True
                        if not unlock:
                            html_temp += f'<td>?</td>'

        elif submit.task.metric == Task.Metric.MAE:
            # Goal:選択
            if goal:
                html_temp += f'<td>MAE <span style="color:#0dcaf0">{submit.task.goal:.1f}</span> 以下 {submit.task.achieveStarHTML(submit.stats)}</td>'

            # invalid判定
            if submit.stats.train < 0:
                if visible_invalid_data:
                    html_temp += '<td>-</td><td>-</td>' if not test else '<td>-</td><td>-</td><td>-</td>'
                else:
                    return ""
                
            else:
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.train, submit.task.goal)}>{submit.stats.train:.3f}(MAE)</td>'
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.valid, submit.task.goal)}>{submit.stats.valid:.3f}(MAE)</td>'
                
                # Test:選択
                if test:
                    if submit.task.type == Task.TaskType.Quest:
                        html_temp += '<td>-</td>'
                    elif submit.task.type == Task.TaskType.Contest:
                        unlock = False
                        if submit.task.achieve(submit.stats):
                            # Questであればいつでも、Contestであれば期間終了後にロック解除
                            if submit.task.afterContest(): 
                                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.test, submit.task.goal)}>{submit.stats.test:.3f}(MAE)</td>'
                                unlock = True
                        if not unlock:
                            html_temp += f'<td>?</td>'

        elif submit.task.metric == Task.Metric.RegistrationRate:
            # Goal:選択
            if goal:
                html_temp += f'<td>データ登録率 <span style="color:#0dcaf0">{submit.task.goal*100:.1f}%</span> 以下 {submit.task.achieveStarHTML(submit.stats)}</td>'

            # invalid判定
            if submit.stats.train < 0:
                if visible_invalid_data:
                    html_temp += '<td>-</td><td>-</td>' if not test else '<td>-</td><td>-</td><td>-</td>'
                else:
                    return ""
                
            else:
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.train, submit.task.goal)}>{submit.stats.train*100:.1f}%(登録率) <a href="/detail/{submit.task.id}/{submit.stats.userid}/{submit.stats.datetime.strftime("%Y%m%d_%H%M%S")}/train" class="link-info">●</a></td>'
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.valid, submit.task.goal)}>{submit.stats.valid*100:.1f}%(登録率) <a href="/detail/{submit.task.id}/{submit.stats.userid}/{submit.stats.datetime.strftime("%Y%m%d_%H%M%S")}/valid" class="link-info">●</a></td>'
                
                # Test:選択
                if test:
                    if submit.task.type == Task.TaskType.Quest:
                        html_temp += '<td>-</td>'
                    elif submit.task.type == Task.TaskType.Contest:
                        unlock = False
                        if submit.task.achieve(submit.stats):
                            # Questであれば存在せず、Contestであれば期間終了後にロック解除
                            if submit.task.afterContest(): 
                                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.test, submit.task.goal)}>{submit.stats.test*100:.1f}(登録率) <a href="/detail/{submit.task.id}/{submit.stats.userid}/{submit.stats.datetime.strftime("%Y%m%d_%H%M%S")}/test" class="link-info">●</a></td>'
                                unlock = True
                        if not unlock:
                            html_temp += f'<td>?</td>'
                            
        elif submit.task.metric == Task.Metric.AverageF1Score:
            # Goal:選択
            if goal:
                html_temp += f'<td>平均F1スコア <span style="color:#0dcaf0">{submit.task.goal:.3f}</span> 以上 {submit.task.achieveStarHTML(submit.stats)}</td>'

            # invalid判定
            if submit.stats.train < 0:
                if visible_invalid_data:
                    html_temp += '<td>-</td><td>-</td>' if not test else '<td>-</td><td>-</td><td>-</td>'
                else:
                    return ""
                
            else:
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.train, submit.task.goal)}>{submit.stats.train:.3f}(F1) <a href="/detail/{submit.task.id}/{submit.stats.userid}/{submit.stats.datetime.strftime("%Y%m%d_%H%M%S")}/train" class="link-info">●</a></td>'
                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.valid, submit.task.goal)}>{submit.stats.valid:.3f}(F1) <a href="/detail/{submit.task.id}/{submit.stats.userid}/{submit.stats.datetime.strftime("%Y%m%d_%H%M%S")}/valid" class="link-info">●</a></td>'
                
                # Test:選択
                if test:
                    if submit.task.type == Task.TaskType.Quest:
                        html_temp += '<td>-</td>'
                    elif submit.task.type == Task.TaskType.Contest:
                        unlock = False
                        if submit.task.achieve(submit.stats):
                            # Questであればいつでも、Contestであれば期間終了後にロック解除
                            if submit.task.afterContest(): 
                                html_temp += f'<td{EvaluatedValueStyle(submit.task.metric, submit.stats.test, submit.task.goal)}>{submit.stats.test:.3f}(F1) <a href="/detail/{submit.task.id}/{submit.stats.userid}/{submit.stats.datetime.strftime("%Y%m%d_%H%M%S")}/test" class="link-info">●</a></td>'
                                unlock = True
                        if not unlock:
                            html_temp += f'<td>?</td>'

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


def CreateInProcHtml(task_id):
    # ユーザ情報を読み込む
    users = User.readUsersCsv(User.USER_CSV_PATH)

    inproc_text = ''
    for user_id in users:
        if os.path.exists(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "user", f"{user_id}_inproc")):
            inproc_text += f"{users[user_id].name} さんの評価を実行中です。<br>"

    return inproc_text + '<br>'


def CreateMyTaskTable(user_id) -> str:
    submits = []

    # user_idのstatsをTaskごとに取得
    for task_id, task in TASK.items():
        task:Task = task
        user_stats = User.getUserStats(task_id)
        if user_id in user_stats:
            stats = user_stats[user_id]

            # 表示最優先の成績を選択
            best_stats = Stats.getBestStats(stats, task)
            if best_stats is not None:
                submit:Submit = Submit(best_stats, task)
                submits.append(submit)

    # 提出日時でソート
    sorted_submits = sorted(submits, key=lambda x: x.stats.datetime, reverse=True)

    # 表のHTMLを作成
    html_table, num_col = CreateRecordTable(sorted_submits, goal=True, test=True,
                                            task_name=True, unlock_mode=UnlockMode.UnlockAll)
    
    return html_table


def CreateSubmitTable(user_id) -> str:
    submits = []

    # user_idのstatsをTaskごとに取得
    for task_id, task in TASK.items():
        stats_temp = User.getUserStats(task_id)
        if user_id in stats_temp:
            for item in stats_temp[user_id]:
                submit: Submit = Submit(item, task)
                submits.append(submit)

    # 提出日時でソート
    sorted_submits = sorted(submits, key=lambda x: x.stats.datetime, reverse=True)
    
    # 表のHTMLを作成
    html_table, num_col = CreateRecordTable(sorted_submits, visible_invalid_data=True,
                                            task_name=True, memo=True,
                                            message=True, unlock_mode=UnlockMode.UnlockAll)

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
    users = User.readUsersCsv(User.USER_CSV_PATH)

    for user_id, user in users.items():
        user_data:User.UserData = user

        # 最新の提出を抽出
        latest_datetime:datetime.datetime = datetime.datetime(1984, 4, 22)
        latest_submit_task:Task = None
        num_submit = 0
        for task_id, task in TASK.items():
            stats_temp = User.getUserStats(task_id)
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
        html_table += f'<td><input type="number" name="goal" value="{task.goal}" step="{0.01 if task.answer_value_type == Task.AnswerValueType.ActiveLearing else 0.1}" class="bg-dark text-white"></td>'
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
    users = User.readUsersCsv(User.USER_CSV_PATH)
    
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
    users = User.readUsersCsv(User.USER_CSV_PATH)
    
    # 認証を行う
    verified = False
    if user_id in users:
        user_data = users[user_id]
        if user_data.key == user_key:
            verified = True
    else:
        user_data = User.UserData()

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
                if task.afterContest():
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
                if value.afterContest():
                    task_list_closed.append(info)
                else:
                    task_list_open.append(info)
        else:
            # スタート前
            if admin:
                task_list_prepare.append(info)
    
    return render_template('index.html',
                           service_name=SETTING["name"]["service"],task_list_open=task_list_open, task_list_closed=task_list_closed, task_list_quest=task_list_quest, task_list_prepare=task_list_prepare, name_contest=SETTING["name"]["contest"],
                           menu=menuHTML(Page.HOME, url_from="/", user_name=user_data.name if verified else ''), admin=admin)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/img'), 'favicon.ico', )


@app.route('/join', methods=['GET', 'POST'])
def join():
    from_url = request.args.get('from')
    if from_url is None:
        from_url = "/"

    if request.method == 'GET':
        return render_template(f'join.html',
                               service_name=SETTING["name"]["service"], from_url=from_url, message="")
    
    elif request.method == 'POST':
        try:
            email = request.form['inputEmail']
            password = request.form['inputPassword']
            password_verify = request.form['inputPasswordVerify']
            next_url = request.form['nextUrl']
        except:
            return render_template(f'join.html', service_name=SETTING["name"]["service"], from_url=from_url, message="入力データを受け取れませんでした。")
        
        # 2つのパスワード入力の一致チェック
        if password != password_verify:
            return render_template(f'join.html', service_name=SETTING["name"]["service"], from_url=from_url, message="再入力したパスワードが一致していません。")
        
        # ユーザ情報を読み込む
        users = User.readUsersCsv(User.USER_CSV_PATH)

        # email重複チェック
        duplicate = False
        for user_id, user_data in users.items():
            if email == user_data.email:
                duplicate = True
                break
        if duplicate:
            Log.write(f"Failed to create account. the email already exist. email: {email}")
            return render_template(f'join.html', service_name=SETTING["name"]["service"], from_url=from_url, message="そのEmail addressは既に登録されています。")
        
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
            return render_template(f'join.html', service_name=SETTING["name"]["service"], from_url=from_url, message="IDを発行できませんでした。")
        
        # パスワードをハッシュ化
        pass_hash = generate_password_hash(password, salt_length=21)
        
        # 本人確認用のキーを作成
        user_key = str(uuid.uuid4()).split('-')[0]

        # ユーザ登録
        name = email.split('@')[0]
        success = User.addUsersCsv(User.USER_CSV_PATH, user_id, email, name, pass_hash, user_key)

        if not success:
            Log.write(f"Failed to create account. addUsersCsv() error. email: {email}")
            return render_template(f'join.html', service_name=SETTING["name"]["service"], from_url=from_url, message="ユーザ情報を登録できませんでした。")

        Log.write(f"Success to create account. email: {email}")

        return render_template(f'user.html', service_name=SETTING["name"]["service"], from_url=from_url, user_email=email, user_id=user_id, user_key=user_key, user_name=name, next_url=next_url, login="true", update_user_data="true")


@app.route('/login', methods=['GET', 'POST'])
def login():
    from_url = request.args.get('from')
    if from_url is None:
        from_url = "/"

    if request.method == 'GET':
        return render_template(f'login.html', service_name=SETTING["name"]["service"], from_url=from_url, message="", email_admin=SETTING["admin"]["email"])
    
    elif request.method == 'POST':
        try:
            email = request.form['inputEmail']
            password = request.form['inputPassword']
            next_url = request.form['nextUrl']
        except:
            return render_template(f'login.html', service_name=SETTING["name"]["service"], from_url=from_url, message="入力データを受け取れませんでした。", email_admin=SETTING["admin"]["email"])
        
        # emailとパスワードで照合
        verified, user_data = VerifyEmailAndPassword(email, password)
        if not verified:
            Log.write(f"Failed to log in. email: {email}")
            return render_template(f'login.html', service_name=SETTING["name"]["service"], from_url=from_url, message="Email addressかPasswordが誤っています。", email_admin=SETTING["admin"]["email"])

        Log.write(f"Success to log in. email: {email}")

        # 認証OKなので本人確認用のキーを作成して渡す
        user_key = str(uuid.uuid4()).split('-')[0]

        # キーを保存
        User.updateUsersCsv(User.USER_CSV_PATH, user_data.id, 'key', user_key)

        # ユーザ情報をクッキーに書き込み
        response = make_response(
            render_template(f'user.html', service_name=SETTING["name"]["service"], from_url=from_url, user_email=email, user_id=user_data.id, user_key=user_key, user_name=user_data.name, next_url=next_url, login="true", update_user_data="true")
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
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    if not verified:
        return redirect(url_for('login'))
    
    # 提出テーブルを作成
    my_task_table_html = CreateMyTaskTable(user_data.id)
    submit_table_html = CreateSubmitTable(user_data.id)

    if request.method == 'GET':
        from_url = request.args.get('from')
        return render_template(
            f'user.html', service_name=SETTING["name"]["service"], login="false",
            user_name=user_data.name, achievement=Markup(User.achievementStrHTML(user_data.id)),
            from_url=from_url if from_url is not None else "/",
            email=user_data.email, user_id=user_data.id, user_key=user_data.key,
            my_task_table=Markup(my_task_table_html), submit_table=Markup(submit_table_html))

    elif request.method == 'POST':
        try:
            user_id = request.form['userID']
            user_key = request.form['userKey']
            verified, user_data = VerifyIdAndKey(user_id, user_key)
            if not verified:
                raise(ValueError())
        except:
            return render_template(f'user.html', service_name=SETTING["name"]["service"], message='ユーザ認証に失敗しました。')
        
        new_name = user_data.name
        message = ''
        try:
            if 'buttonChangeName' in request.form:
                # 名前を変更
                new_name = request.form['newName']
                success = User.updateUsersCsv(User.USER_CSV_PATH, user_id, 'name', new_name)
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
                success = User.updateUsersCsv(User.USER_CSV_PATH, user_id, 'pass_hash', pass_hash)
                if not success:
                    message = 'ユーザ情報の更新に失敗しました。'
                    raise(ValueError())
                message = 'パスワードを変更しました。'
                Log.write(f"Success to change password. user_id: {user_id}")

        except:
            return render_template(f'user.html',
                                    service_name=SETTING["name"]["service"], 
                                    user_name=user_data.name, message=message, achievement=Markup(User.achievementStrHTML(user_data.id)),
                                    email=user_data.email, user_id=user_data.id, user_key=user_data.key,
                                    my_task_table=Markup(my_task_table_html), submit_table=Markup(submit_table_html))

        return render_template(f'user.html',
                                service_name=SETTING["name"]["service"], 
                                user_name=new_name, message=message, achievement=Markup(User.achievementStrHTML(user_data.id)),
                                email=user_data.email, user_id=user_data.id, user_key=user_data.key,
                                my_task_table=Markup(my_task_table_html), submit_table=Markup(submit_table_html))


@app.route('/source/<task_id>/<filename>')
def source(task_id, filename):
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    if not verified:
        return redirect(url_for('login'))

    file_path = os.path.join(Task.TASKS_DIR, task_id, Task.USER_MODULE_DIR_NAME, filename)
    if not os.path.exists(file_path):
        return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='ファイルが見つかりません')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 元のファイル名を復元
        filename_split = filename.split('_')
        filename_head = f"{filename_split[0]}_{filename_split[1]}_{filename_split[2]}_{filename_split[3]}_"
        filename_org = filename.replace(filename_head, '')
    except Exception as e:
        return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='ファイルを読み込めません')

    return render_template(f'source.html', service_name=SETTING["name"]["service"], source=content, filename=filename_org)


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
        with open(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, Task.TIMESTAMP_FILE_NAME), "r", encoding='utf-8') as f:
            timestamp = f.read()
    except:
        timestamp = ''

    return timestamp


@app.route('/<task_id>/card.png')
def get_taskcard(task_id):
    return send_from_directory(os.path.join(Task.TASKS_DIR, task_id), "card.png")


@app.route('/<task_id>/resource/<filename>')
def get_task_image(task_id, filename):
    return send_from_directory(os.path.join(Task.TASKS_DIR, task_id, Task.RESOURCE_DIR_NAME), filename)


@app.route("/<task_id>/task")
def task(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))

    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)

    # タスク情報を読み込む
    task:Task = Task(task_id)

    return render_template(f'tasks/{task_id}/index.html',
                           menu=menuHTML(Page.TASK, task_id, url_from=f"/{task_id}/task", admin=admin, user_name=user_data.name if verified else ''),
                           service_name=SETTING["name"]["service"],
                           task_name=task.dispname(SETTING["name"]["contest"]), task_id=task_id,
                           goal=Task.goalText(task.metric, task.goal))


@app.route("/<task_id>/board", methods=['GET'])
def board(task_id):
    if not task_id in TASK:
        return redirect(url_for('index'))

    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    
    # タスク情報を読み込む
    task:Task = Task(task_id)

    # ユーザ成績を読み込む
    user_stats = User.getUserStats(task_id)

    # ユーザごとに表示最優先の成績を選択
    best_stats_every_user = []
    best_stats_in_contest = []
    my_stats = None
    for user_name, stats in user_stats.items():
        best_stats:Stats = Stats.getBestStats(stats, task)
        if best_stats is not None:
            best_stats_every_user.append(best_stats)
            if verified and best_stats.userid == user_data.id:
                my_stats = best_stats

        # コンテスト期間中の各ユーザベスト
        if task.afterContest():
            stats_in_contest = []
            for item in stats:
                one_stats:Stats = item
                if one_stats.datetime >= task.start_date and one_stats.datetime < task.end_date:
                    stats_in_contest.append(one_stats)
            best_stats:Stats = Stats.getBestStats(stats_in_contest, task)
            if best_stats is not None:
                best_stats_in_contest.append(best_stats)

    # 日付順にソート
    sorted_stats_list = sorted(best_stats_every_user, key=lambda x: x.datetime, reverse=True)
    submit_list = []
    for stats in sorted_stats_list:
        submit_list.append(Submit(stats, task))
    
    # test成績順にソート
    if len(best_stats_in_contest) > 0:
        sorted_stats_list_in_contest = sorted(best_stats_in_contest, key=lambda x: x.test, reverse=True if task.metric == Task.Metric.Accuracy else False)
        submit_list_in_contest = []
        for stats in sorted_stats_list_in_contest:
            submit_list_in_contest.append(Submit(stats, task))
 
    # unlock判定
    unlock = False
    if verified and my_stats is not None:
        # ユーザの認証ができている場合、このタスクの目標を達成しているか確認
        if task.achieve(my_stats):
            # Questであればいつでも、Contestであれば期間終了後にロック解除
            if task.type == Task.TaskType.Quest:
                unlock = True
            elif task.afterContest(): 
                unlock = True

    # 表を作成
    html_table, num_col = CreateRecordTable(submit_list, table_id='sortable-table', user_name=True,
                                            test=True if task.type == Task.TaskType.Contest else False,
                                            unlock_mode=UnlockMode.UnlockAchieveStats if unlock else UnlockMode.LockAll)

    # コンテスト終了時の成績表を作成
    html_contest_result = None
    if len(best_stats_in_contest) > 0:
        html_contest_result, num_col = CreateRecordTable(submit_list_in_contest, user_name=True, test=True,
                                                         unlock_mode=UnlockMode.UnlockAchieveStats if unlock else UnlockMode.LockAll)

    return render_template('board.html',
                           service_name=SETTING["name"]["service"], 
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           table_board=Markup(html_table),
                           table_contest_result=Markup(html_contest_result) if html_contest_result is not None else None,
                           menu=menuHTML(Page.BOARD, task_id, url_from=f"/{task_id}/board", admin=admin, user_name=user_data.name if verified else ''),
                           inproc_text=Markup(CreateInProcHtml(task_id)),
                           goal=Task.goalText(task.metric, task.goal),
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
    user_stats = User.getUserStats(task_id)

    # 辞書をリスト化してソート
    unlock = False
    stats_list = []
    for stats in user_stats.values():
        for item in stats:
            stats_list.append(item)
            if verified and item.userid == user_data.id and task.achieve(item):
                # Questであればいつでも、Contestであれば期間終了後にロック解除
                if task.type == Task.TaskType.Quest:
                    unlock = True
                elif task.afterContest(): 
                    unlock = True

    sorted_stats_list = sorted(stats_list, key=lambda x: x.datetime, reverse=True)
    submit_list = []
    for stats in sorted_stats_list:
        submit_list.append(Submit(stats, task))

    # 表を作成
    html_table, num_col = CreateRecordTable(submit_list, table_id='sortable-table', user_name=True,
                                            test=True if task.type == Task.TaskType.Contest else False,
                                            unlock_mode=UnlockMode.UnlockAchieveStats if unlock else UnlockMode.LockAll)

    return render_template('log.html',
                           service_name=SETTING["name"]["service"], 
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           table_log=Markup(html_table),
                           menu=menuHTML(Page.LOG, task_id, url_from=f"/{task_id}/log", admin=admin, user_name=user_data.name if verified else ''), 
                           inproc_text=Markup(CreateInProcHtml(task_id)), 
                           num_col=num_col,
                           task_id=task_id,
                           goal=Task.goalText(task.metric, task.goal))


@app.route('/<task_id>/upload', methods=['GET', 'POST'])
def upload_file(task_id):
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    if not verified:
        return redirect(url_for('login') + f'?from=/{task_id}/upload')

    if not task_id in TASK:
        return redirect(url_for('index'))
    
    task:Task = TASK[task_id]
    
    if task.suspend:
        return redirect(url_for('index'))
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    msg = ""
    # 提出可否チェック
    upload_enabled = True
    stats = User.getUserStats(task_id)
    if user_data.id in stats:
        sorted_submits = sorted(stats[user_data.id], key=lambda x: x.datetime, reverse=True) # 提出日時でソート
        if sorted_submits[0].datetime + datetime.timedelta(minutes=task.submit_interval_minutes) > datetime.datetime.now():
            last_minutes = (sorted_submits[0].datetime + datetime.timedelta(minutes=task.submit_interval_minutes) - datetime.datetime.now()).total_seconds() / 60
            msg = f"提出可能になるのは、前回の提出から一定期間後となります。(あと{math.ceil(last_minutes)}分ほどお待ちください)"
            upload_enabled = False

    if os.path.exists(os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "user", f"{user_data.id}_inproc")):
        msg = f"現在、評価を実行中です。提出可能になるのは、評価完了かつ提出から一定期間後となります。"
        upload_enabled = False

    py_files = glob.glob(os.path.join(Task.TASKS_DIR, task_id, Task.UPLOAD_DIR_NAME, user_data.id, '*.py'))
    if len(py_files) > 0:
        msg = f"現在、評価待ち中です。提出可能になるのは、評価完了かつ提出から一定期間後となります。"
        upload_enabled = False

    if request.method == 'POST' and upload_enabled:
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
                if 'attachment' in request.files and request.files['attachment'].filename != '':
                    # 添付ファイルがある場合
                    attachment = request.files['attachment']
                else:
                    attachment = None

                try:
                    save_dir = os.path.join(Task.TASKS_DIR, task_id, Task.UPLOAD_DIR_NAME, user_id)
                    new_filename = secure_filename(file.filename)

                    # まだディレクトリが存在しなければ作成(Taskのディレクトリがなければそれも生成)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    if os.path.exists(save_dir):
                        # 添付ファイルがある場合は保存
                        if attachment is not None:
                            attachment_filename = secure_filename(attachment.filename)
                            attachment.save(os.path.join(save_dir, new_filename + '.attachment'))
                            msg += f' 添付ファイル {attachment_filename} がアップロードされました。'

                        # メモを保存
                        if request.form['memo'] != "":
                            with open(os.path.join(save_dir, new_filename + '.txt'), mode='w', encoding='utf-8') as f:
                                f.write(request.form['memo'])

                        # .pyファイルを保存(これは最後に行う。評価システム側ではこのファイルが存在する場合にアップロード完了と判断するため)
                        file.save(os.path.join(save_dir, new_filename))
                        msg = f'{file.filename} がアップロードされました。'
                    else:
                        raise(ValueError("アップロード先のディレクトリが存在しません。"))
                except Exception as e:
                    msg = "アップロードに失敗しました。"
                    print(e)

    return render_template('upload.html',
                           task_id=task_id, task_name=task.name, message=msg,
                           user_id=user_data.id, user_key=user_data.key,
                           menu=menuHTML(Page.UPLOAD, task_id, url_from=f"/{task_id}/upload", admin=admin, user_name=user_data.name if verified else ''),
                           service_name=SETTING["name"]["service"],
                           url_from=f"/{task_id}/upload", time_limit=task.timelimit_per_data,
                           attachment=task.attachment,
                           upload_enabled=upload_enabled)
  

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
    user_stats = User.getUserStats(task_id)

    # 辞書をリスト化してソート
    stats_list = []
    for stats in user_stats.values():
        for item in stats:
            stats_list.append(item)
    sorted_stats_list = sorted(stats_list, key=lambda x: x.datetime, reverse=True)
    submit_list = []
    for stats in sorted_stats_list:
        submit_list.append(Submit(stats, task))

    # 表を作成
    html_table, num_col = CreateRecordTable(submit_list, visible_invalid_data=True, table_id='sortable-table', user_name=True,
                                            test=True if task.type == Task.TaskType.Contest else False,
                                            message=True, unlock_mode=UnlockMode.UnlockAll)

    return render_template('log.html',
                           task_id=task_id,
                           service_name=SETTING["name"]["service"],
                           task_name=task.dispname(SETTING["name"]["contest"]),
                           table_log=Markup(html_table),
                           menu=menuHTML(Page.ADMIN, task_id, url_from=f"/{task_id}/admin", admin=admin, user_name=user_data.name if verified else ''),
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
                           service_name=SETTING["name"]["service"],
                           user_table=Markup(CreateUserTable()),
                           task_table=Markup(CreateTaskTable(TASK)),
                           log_table=Markup(Log.createTable()))


@app.route('/detail/<task_id>/<user_id>/<dt>/<data_type>')
def detail(task_id, user_id, dt, data_type):
    # ユーザ認証
    verified, user_data, admin = VerifyByCookie(request)
    if not verified:
        return redirect(url_for('login'))
    
    # タスク情報を読み込む
    task:Task = Task(task_id)

    if task.metric != Task.Metric.RegistrationRate and task.metric != Task.Metric.AverageF1Score:
        return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='詳細を表示できるTaskではありません')
    
    if data_type != "train" and data_type != "valid" and data_type != "test":
        return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='データタイプの指定が不正です')

    if data_type == "test":
        if not task.afterContest():
            return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='まだ表示できません')

    file_path = os.path.join(Task.TASKS_DIR, task_id, Task.OUTPUT_DIR_NAME, "detail", f"{user_id}_{dt}.csv")
    if not os.path.exists(file_path):
        return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='詳細を表示するファイルが見つかりません')

    # アクティブラーニングTaskの詳細表示
    if task.metric == Task.Metric.RegistrationRate:
        class_name = {}
        try:
            dataset_path = os.path.join(Task.TASKS_DIR, task_id, data_type, FILENAME_DATASET_JSON)
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            for key, value in dataset["settings"]["class_name"].items():
                class_name[int(key)] = value
        except Exception as e:
            print(e)

        # detal csvの読み込み
        try:        
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                all_rows = list(reader)
                detail = {}
                iRow = 0
                stats = {}
                while iRow < len(all_rows):
                    # 最初のtrain/valid/testがTaskの成績
                    if not "train" in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == "train":
                        stats["train"] = float(all_rows[iRow][2])
                    if not "valid" in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == "valid":
                        stats["valid"] = float(all_rows[iRow][2])
                    if not "test" in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == "test":
                        stats["test"] = float(all_rows[iRow][2])

                    # detail記述の始まり
                    if len(all_rows[iRow]) >= 2 and all_rows[iRow][0] == "detail":
                        kind = all_rows[iRow][1]
                        detail[kind] = {"precision":[], "recall":[]} #P/RのList
                        iRow += 1 #ヘッダ行を飛ばす
                        for iRR in range(101): #0..100%
                            iRow += 1
                            while all_rows[iRow] and all_rows[iRow][-1] == '':
                                all_rows[iRow].pop() #末尾の空白要素を削除
                            detail[kind]["precision"].append(list(map(float, all_rows[iRow][1:])))
                        iRow += 1 #ヘッダ行を飛ばす
                        for iRR in range(101): #0..100%
                            iRow += 1
                            while all_rows[iRow] and all_rows[iRow][-1] == '':
                                all_rows[iRow].pop() #末尾の空白要素を削除
                            detail[kind]["recall"].append(list(map(float, all_rows[iRow][1:])))

                    iRow += 1
        except Exception as e:
            print(f"{e}")
            return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='ファイルを読み込めません')
        
        # データが存在しなければ非表示
        if not data_type in detail:
            return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='詳細データが存在しません')

        # ゴール達成していなければ非表示
        if data_type == "test":
            if stats["train"] > task.goal or stats["valid"] > task.goal or stats["test"] > task.goal:
                return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='Goalを達成した場合のみ表示可能です')
        
        # グラフ表示データ
        def dataDict(data, precision=True):
            cmap = plt.get_cmap('tab20')
            data_dict = {
                "labels": [], "datasets": []
            }
            for iRR in range(data.shape[0]):
                data_dict["labels"].append(f"{iRR}")
            for iClass in range(data.shape[1]):
                class_dict = {"label":class_name[iClass] if iClass in class_name else f"class{iClass}", "data":[], "borderColor":matplotlib.colors.rgb2hex(cmap(iClass))}
                for iData in range(data.shape[0]):
                    class_dict["data"].append(data[iData][iClass])
                data_dict["datasets"].append(class_dict)
            return data_dict
        
        def optionsDict(precision=True):
            config_dict = {"scales":{"x":{},"y":{}}, "elements":{"point":{"radius":0}}}
            config_dict["scales"]["x"]["beginAtZero"] = True
            config_dict["scales"]["x"]["title"] = {
                "display": True,
                "text": "データ登録率 [%]"
            }
            config_dict["scales"]["y"]["beginAtZero"] = True
            config_dict["scales"]["y"]["title"] = {
                "display": True,
                "text": "Precision" if precision else "Recall"
            }
            config_dict["scales"]["y"]["grid"] = {
                "color": "#555"
            }
            return config_dict

        precision_data = dataDict(np.array(detail[data_type]["precision"]), precision=True)
        recall_data = dataDict(np.array(detail[data_type]["recall"]), precision=False)

        return render_template(f'detail_active-learning.html',
                            service_name=SETTING["name"]["service"],
                            user_name=User.userIDtoUserName(user_id),
                            data_type=data_type,
                            stats=f"{stats[data_type]*100:.01f}",
                            train_precision=precision_data,
                            train_recall=recall_data,
                            options_precision=optionsDict(True),
                            options_recall=optionsDict(False))
    
    # 特徴抽出タスクの詳細表示
    elif task.metric == Task.Metric.AverageF1Score:
        # detal csvの読み込み
        if not os.path.exists(file_path):
            return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='詳細を表示するファイルが見つかりません')

        try:        
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                all_rows = list(reader)
                detail = {}
                iRow = 0
                stats = {}
                each_stats = {}
                while iRow < len(all_rows):
                    # 最初のtrain/valid/testがTaskの成績
                    if not "train" in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == "train":
                        stats["train"] = float(all_rows[iRow][2])
                        iRow += 1
                    if not "valid" in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == "valid":
                        stats["valid"] = float(all_rows[iRow][2])
                        iRow += 1
                    if not "test" in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == "test":
                        stats["test"] = float(all_rows[iRow][2])
                        iRow += 1

                    # 2つ目以降の各行が各データタイプの詳細
                    if data_type in stats and len(all_rows[iRow]) >= 3 and all_rows[iRow][0] == data_type:
                        each_stats[int(all_rows[iRow][1])] = float(all_rows[iRow][2])

                    # detail記述の始まり
                    if len(all_rows[iRow]) >= 2 and all_rows[iRow][0] == "detail":
                        kind = all_rows[iRow][1]
                        task_index = int(all_rows[iRow][2])
                        if not kind in detail:
                            detail[kind] = {}
                        detail[kind][task_index] = {}
                        iRow += 2 #ヘッダ行を飛ばす

                        # カテゴリ名リストを取得
                        try:
                            dataset_path = os.path.join(Task.TASKS_DIR, task_id, data_type, FILENAME_DATASET_JSON)
                            with open(dataset_path, 'r', encoding='utf-8') as f:
                                dataset = json.load(f)
                            class_name = {}
                            for key, value in dataset["data"][task_index]["class_name"].items():
                                class_name[int(key)] = value
                        except:
                            class_name = {}

                        while len(all_rows[iRow]) >= 8:
                            row = all_rows[iRow]
                            class_index = int(row[0])
                            gt = int(row[1])
                            tp = float(row[2])
                            fn = float(row[3])
                            fp = float(row[4])
                            precision = float(row[5])
                            recall = float(row[6])
                            f1_score = float(row[7])
                            detail[kind][task_index][class_index] = {
                                "class_name": class_name[class_index] if class_index in class_name else f"class{class_index}",
                                "gt": gt,
                                "tp": tp,
                                "fn": fn,
                                "fp": fp,
                                "precision": precision,
                                "recall": recall,
                                "f1_score": f1_score,
                            }
                            iRow += 1
                    iRow += 1
        except Exception as e:
            print(f"{e}")
            return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='ファイルを読み込めません')
        
        # データが存在しなければ非表示
        if not data_type in detail:
            return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='詳細データが存在しません')

        # ゴール達成していなければ非表示
        if data_type == "test":
            if stats["train"] < task.goal or stats["valid"] < task.goal or stats["test"] < task.goal:
                return render_template(f'source.html', service_name=SETTING["name"]["service"], filename='Goalを達成した場合のみ表示可能です')
            
        return render_template(f'detail_feature-extraction.html',
                            service_name=SETTING["name"]["service"],
                            user_name=User.userIDtoUserName(user_id),
                            data_type=data_type,
                            stats=stats[data_type],
                            each_stats=each_stats,
                            detail=detail[data_type])


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
    app.run(debug=True, host='0.0.0.0', port=80)
