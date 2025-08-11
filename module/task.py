import os
import datetime
from enum import Enum
import json
from functools import cmp_to_key
from markupsafe import Markup
import glob
import shutil


class Task:
    TASKS_DIR =r"tasks"
    FILENAME_TASK_JSON = r"task.json"
    BACKUP_DIR_NAME = r"backup"
    OUTPUT_DIR_NAME = r"output"
    USER_RESULT_DIR_NAME = "user"
    UPLOAD_DIR_NAME = r"upload"
    USER_MODULE_DIR_NAME = r"user_module"
    TIMESTAMP_FILE_NAME = r"timestamp.txt"
    RESOURCE_DIR_NAME = r"resource"

    class AnswerValueType(Enum):
        real = 1
        integer = 2
        Image1ch = 3
        Image3ch = 4
        ActiveLearing = 5
        FeatureExtraction = 6

    class DataType(Enum):
        train = 1
        valid = 2
        test = 3

    class Metric(Enum):
        Accuracy = 1
        MAE = 2
        RegistrationRate = 3
        AverageF1Score = 4

    class InputDataType(Enum):
        Image1ch = 1
        Image3ch = 2
        Vector = 3

    class TaskType(Enum):
        Quest = 1
        Contest = 2

    class ParameterType(Enum):
        real = 1
        integer = 2

    id: str
    name: str
    explanation: str
    start_date: datetime
    end_date: datetime
    answer_value_type: AnswerValueType
    metric: Metric
    input_data_type: InputDataType
    multi_input_data: bool
    attachment: bool
    type: TaskType = TaskType.Quest
    goal = 0
    timelimit_per_data: float = 1.0
    suspend: bool = False

    def __init__(self, task_id) -> None:
        try:
            task_json_path = os.path.join(Task.TASKS_DIR, task_id, Task.FILENAME_TASK_JSON)
            json_open = open(task_json_path, 'r', encoding='utf-8')
            task = json.load(json_open)
            task = task["info"]

            self.id = task["id"]
            self.name = task["name"]
            self.explanation = task["explanation"]
            self.start_date = datetime.datetime.strptime(task["start_date"], '%Y-%m-%d')
            self.end_date = datetime.datetime.strptime(task["end_date"], '%Y-%m-%d')
            self.answer_value_type = self.answerValueType(task["answer_value_type"])
            self.metric = self.metricType(task["metric"])
            self.input_data_type = self.inputDataType(task["input_data_type"])
            self.multi_input_data = task["multi_input_data"]
            self.attachment = task["attachment"] if "attachment" in task else False
            self.submit_interval_minutes = task["submit_interval_minutes"] if "submit_interval_minutes" in task else 0
            if "type" in task:
                if task["type"] == "quest":
                    self.type = Task.TaskType.Quest
                elif task["type"] == "contest":
                    self.type = Task.TaskType.Contest
            if "goal" in task:
                self.goal = float(task["goal"])
            if "timelimit_per_data" in task:
                self.timelimit_per_data = float(task["timelimit_per_data"])
            if "suspend" in task:
                self.suspend = task["suspend"]
        except Exception as e:
            print(e)
            print(f"Taskを読み込めませんでした: {task_id}")

    @staticmethod
    def readTasks():
        tasks = {}

        # タスク一覧を作成
        dir_list = glob.glob(Task.TASKS_DIR + '/**/')
        for dir in dir_list:
            # ディレクトリ名を取得→タスクIDとして使う
            task_id = os.path.basename(os.path.dirname(dir))

            try:
                # タスク情報を取得
                task:Task = Task(task_id)
                print(f"found task: ({task_id}) {task.name}")
                tasks[task_id] = task
            except:
                continue

        return tasks

    @staticmethod
    def answerValueType(answer_value_type):
        if answer_value_type == "integer":
            return Task.AnswerValueType.integer
        elif answer_value_type == "real":
            return Task.AnswerValueType.real
        elif answer_value_type == "image-1ch":
            return Task.AnswerValueType.Image1ch
        elif answer_value_type == "image-3ch":
            return Task.AnswerValueType.Image3ch
        elif answer_value_type == "integer-list":
            return Task.AnswerValueType.IntegerList
        elif answer_value_type == "ActiveLearing":
            return Task.AnswerValueType.ActiveLearing
        elif answer_value_type == "FeatureExtraction":
            return Task.AnswerValueType.FeatureExtraction
        else:
            raise(ValueError("無効なanswer_value_type指定です。"))

    @staticmethod
    def metricType(metric:str) -> Metric:
        if metric == "Accuracy":
            return Task.Metric.Accuracy
        elif metric == "MAE":
            return Task.Metric.MAE
        elif metric == "RegistrationRate":
            return Task.Metric.RegistrationRate
        elif metric == "AverageF1Score":
            return Task.Metric.AverageF1Score
        else:
            raise(ValueError("無効なmetric指定です。"))

    @staticmethod
    def inputDataType(input_data_type:str) -> InputDataType:
        if input_data_type == "image-1ch":
            return Task.InputDataType.Image1ch
        elif input_data_type == "image-3ch":
            return Task.InputDataType.Image3ch
        elif input_data_type == "vector":
            return Task.InputDataType.Vector
        else:
            raise(ValueError("無効なinput_data_type指定です。"))

    @staticmethod
    def goalText(metric:Metric, goal):
        if metric == Task.Metric.Accuracy:
            goal_text = f'正解率 <span style="color:#0dcaf0">{goal*100:.1f}</span> % 以上'
        elif metric == Task.Metric.MAE:
            goal_text = f'平均絶対誤差 <span style="color:#0dcaf0">{goal}</span> 以下'
        elif metric == Task.Metric.RegistrationRate:
            goal_text = f'データ登録率 <span style="color:#0dcaf0">{goal*100}%</span> 以下'
        elif metric == Task.Metric.AverageF1Score:
            goal_text = f'平均F値 <span style="color:#0dcaf0">{goal}</span> 以上'
        return Markup(goal_text)

    def afterContest(self):
        if self.type == Task.TaskType.Contest and datetime.datetime.now() >= self.end_date:
            return True
        else:
            return False

    def dispname(self, name_contest:str) -> str:
        task_name = f'{self.name} - '
        if self.type == Task.TaskType.Contest:
            task_name += name_contest
            if datetime.datetime.now() >= self.start_date and datetime.datetime.now() < self.end_date:
                task_name += f'開催中({self.start_date.strftime("%Y-%m-%d")}～{(self.end_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")})'
        elif self.type == Task.TaskType.Quest:
            task_name += "Quest"

        return task_name
    
    def save(self) -> bool:
        try:
            json_path = os.path.join(Task.TASKS_DIR, self.id, Task.FILENAME_TASK_JSON)

            # バックアップ
            if not os.path.exists(os.path.join(Task.TASKS_DIR, self.id)):
                raise(ValueError(f"Task{self.id}のディレクトリが存在しません。"))
            backup_dir = os.path.join(Task.TASKS_DIR, self.id, Task.BACKUP_DIR_NAME)
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            shutil.copy2(json_path, os.path.join(backup_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + Task.FILENAME_TASK_JSON))

            # 現在の設定を読み込む
            json_open = open(json_path, 'r', encoding='utf-8')
            task = json.load(json_open)

            # 設定を書き換える(書き換え可能なものだけ)
            task["info"]["start_date"] = self.start_date.strftime("%Y-%m-%d")
            task["info"]["end_date"] = self.end_date.strftime("%Y-%m-%d")
            task["info"]["goal"] = self.goal
            task["info"]["timelimit_per_data"] = self.timelimit_per_data
            task["info"]["suspend"] = self.suspend

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(task, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(e)
            return False
        
        return True

    def achieve(self, stats) -> bool:
        # 不適正チェック
        if stats.train < 0 or stats.valid < 0 or (self.type == Task.TaskType.Contest and stats.test < 0):
            return False

        achieve = True

        if self.metric == Task.Metric.Accuracy:
            if stats.train < self.goal or stats.valid < self.goal or (self.type == Task.TaskType.Contest and stats.test < self.goal):
                achieve = False

        elif self.metric == Task.Metric.MAE:
            if stats.train > self.goal or stats.valid > self.goal or (self.type == Task.TaskType.Contest and stats.test > self.goal):
                achieve = False

        elif self.metric == Task.Metric.RegistrationRate:
            if stats.train > self.goal or stats.valid > self.goal or (self.type == Task.TaskType.Contest and stats.test > self.goal):
                achieve = False

        elif self.metric == Task.Metric.AverageF1Score:
            if stats.train < self.goal or stats.valid < self.goal or (self.type == Task.TaskType.Contest and stats.test < self.goal):
                achieve = False

        return achieve


    def achieveStarHTML(self, stats):
        # コンテスト中の場合は非表示
        if self.type == Task.TaskType.Contest and datetime.datetime.now() < self.end_date:
            return ''
        
        if not self.achieve(stats):
            return ''

        return f'<span style="color:#0dcaf0">{"★" if self.type == Task.TaskType.Contest else "☆"}</span>'


    @staticmethod
    def achieveValue(metric:Metric, evaluated_value, goal) -> bool:
        achieve = False
        if metric == Task.Metric.Accuracy:
            if evaluated_value >= goal:
                achieve = True
        elif metric == Task.Metric.MAE:
            if evaluated_value <= goal:
                achieve = True
        elif metric == Task.Metric.RegistrationRate:
            if evaluated_value <= goal:
                achieve = True
        elif metric == Task.Metric.AverageF1Score:
            if evaluated_value >= goal:
                achieve = True

        return achieve


class Stats:
    userid : str
    datetime : datetime
    filename : str = ""
    train : list = []
    valid : list = []
    test : list = []
    message : str = ''
    memo : str = ''
    metric: Task.Metric
    goal: float

    # ユーザ成績のcsvファイルに書かれた1行の成績記録をもとにstatsを読み取る
    def __init__(self, user_stats_line:str, metric:Task.Metric, goal:float, userid) -> None:
        self.userid = userid
        self.metric = metric
        self.goal = goal

        raw = user_stats_line.rstrip(os.linesep).split(",")
        
        try:
            self.datetime = datetime.datetime.strptime(raw[0] + " " + raw[1], "%Y/%m/%d %H:%M:%S")
        except:
            self.datetime = datetime.datetime(1984, 4, 22)

        try:
            self.filename = raw[2]
        except:
            self.filename = ''

        if metric == Task.Metric.Accuracy:
            try:
                self.train = float(raw[5])
            except:
                self.train = -1

            try:
                self.valid = float(raw[8])
            except:
                self.valid = -1

            try:
                self.test = float(raw[11])
            except:
                self.test = -1

            try:
                self.message = raw[12]
            except:
                self.message = ''

            try:
                self.memo = raw[13]
            except:
                self.memo = ''

        elif metric == Task.Metric.MAE or metric == Task.Metric.RegistrationRate or metric == Task.Metric.AverageF1Score:
            try:
                self.train = float(raw[3])
            except:
                self.train = -1

            try:
                self.valid = float(raw[4])
            except:
                self.valid = -1

            try:
                self.test = float(raw[5])
            except:
                self.test = -1

            try:
                self.message = raw[6]
            except:
                self.message = ''

            try:
                self.memo = raw[7]
            except:
                self.memo = ''


    @staticmethod
    def getBestStats(stats:list, task:Task):
        def compare(a:Stats, b:Stats):
            goal = a.goal
            train_a = a.train
            train_b = b.train
            valid_a = a.valid
            valid_b = b.valid
            if a.test >= 0:
                test = True
                test_a = a.test
                test_b = b.test
            else:
                test = False

            if a.metric == Task.Metric.MAE or a.metric == Task.Metric.RegistrationRate:
                # 低い方がよい値となるよう反転させる
                goal *= -1
                train_a *= -1
                train_b *= -1
                valid_a *= -1
                valid_b *= -1
                if test:
                    test_a *= -1
                    test_b *= -1
            
            # Goal達成チェック
            train_a_achieve = True if train_a >= goal else False
            train_b_achieve = True if train_b >= goal else False
            valid_a_achieve = True if valid_a >= goal else False
            valid_b_achieve = True if valid_b >= goal else False
            if test:
                test_a_achieve = True if test_a >= goal else False
                test_b_achieve = True if test_b >= goal else False

            # test>valid>trainの順で目標達成していること
            if test:
                if test_a_achieve and not test_b_achieve:
                    return 1 #a
                if not test_a_achieve and test_b_achieve:
                    return -1 #b
                
            if valid_a_achieve and not valid_b_achieve:
                return 1 #a
            if not valid_a_achieve and valid_b_achieve:
                return -1 #b
            
            if train_a_achieve and not train_b_achieve:
                return 1 #a
            if not train_a_achieve and train_b_achieve:
                return -1 #b
            
            # test>valid>trainの順で性能が高い方を優先
            if test:
                if test_a > test_b:
                    return 1 #a
                if test_a < test_b:
                    return -1 #b
                
            if valid_a > valid_b:
                return 1 #a
            if valid_a < valid_b:
                return -1 #b
            
            if train_a > train_b:
                return 1 #a
            if train_a < train_b:
                return -1 #b
            
            # 提出日時が新しい方を優先
            if a.datetime > b.datetime:
                return 1 #a
            if a.datetime < b.datetime:
                return -1 #b
            
            return 0 # 引き分け
        
        def compare_without_test(a:Stats, b:Stats):
            goal = a.goal
            train_a = a.train
            train_b = b.train
            valid_a = a.valid
            valid_b = b.valid

            if a.metric == Task.Metric.MAE or a.metric == Task.Metric.RegistrationRate:
                # 低い方がよい値となるよう反転させる
                goal *= -1
                train_a *= -1
                train_b *= -1
                valid_a *= -1
                valid_b *= -1
            
            # Goal達成チェック
            train_a_achieve = True if train_a >= goal else False
            train_b_achieve = True if train_b >= goal else False
            valid_a_achieve = True if valid_a >= goal else False
            valid_b_achieve = True if valid_b >= goal else False

            # test>valid>trainの順で目標達成していること
            if valid_a_achieve and not valid_b_achieve:
                return 1 #a
            if not valid_a_achieve and valid_b_achieve:
                return -1 #b
            
            if train_a_achieve and not train_b_achieve:
                return 1 #a
            if not train_a_achieve and train_b_achieve:
                return -1 #b
            
            # test>valid>trainの順で性能が高い方を優先
            if valid_a > valid_b:
                return 1 #a
            if valid_a < valid_b:
                return -1 #b
            
            if train_a > train_b:
                return 1 #a
            if train_a < train_b:
                return -1 #b
            
            # 提出日時が新しい方を優先
            if a.datetime > b.datetime:
                return 1 #a
            if a.datetime < b.datetime:
                return -1 #b
            
            return 0 # 引き分け

        in_contest = True if task.type == Task.TaskType.Contest and datetime.datetime.now() < task.end_date else False

        target_stats = []
        for item in stats:
            if item.train < 0 or item.valid < 0:
                continue
            target_stats.append(item)

        if len(target_stats) == 0:
            return None

        try:
            if in_contest:
                sorted_list = sorted(target_stats, key=cmp_to_key(compare_without_test))
            else:
                sorted_list = sorted(target_stats, key=cmp_to_key(compare))
            best_stats = sorted_list[-1]
        except:
            return None

        return best_stats


class Log:
    LOG_DIR = r"./data/log"

    @staticmethod
    def write(log:str) -> bool:
        if not os.path.exists(Log.LOG_DIR):
            os.makedirs(Log.LOG_DIR)

        now = datetime.datetime.now()
        log_file_neme = now.strftime('log_%Y%m%d.log')

        success = False
        for i in range(1000):
            try:
                with open(os.path.join(Log.LOG_DIR, log_file_neme), "a", encoding='utf-8') as f:
                    f.write(now.strftime('%Y-%m-%d %H:%M:%S,'))
                    f.write(log + '\n')
                    success = True
            except:
                print("Retry write log")
                continue
            break

        return success


    @staticmethod
    def createTable() -> str:
        # 表のHTMLを作成
        html_table = '<table class="table table-dark">'
        html_table += "<thead><tr>"
        html_table += "<th>Datetime</th>"
        html_table += "<th>Log</th>"
        html_table += "</tr></thead>"
        html_table += "<tbody>"

        paths = glob.glob(os.path.join(Log.LOG_DIR, "*.log"))
        paths = sorted(paths)
        for path in paths:
            lines = []
            try:
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()
            except:
                print(f"cannot open {path}")

            for line in lines:
                datetime = line.split(',')[0]
                html_table += f"<tr><td>{datetime}</td><td>{line.replace(datetime+',', '')}</td></tr>"

        html_table += "</tbody>"
        html_table += "</table>"

        return html_table
        