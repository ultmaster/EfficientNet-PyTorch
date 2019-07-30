import json
import os
import re
import torch
from argparse import ArgumentParser
from datetime import timedelta, datetime

from dateutil import parser as dateutil_parser
import xlsxwriter
from tensorboardX import SummaryWriter


def analyze_metrics(root_dir):
    categories = ["task_block_in_byte", "task_block_out_byte", "task_cpu_percent",
                  "task_gpu_mem_percent", "task_gpu_percent", "task_mem_usage_byte",
                  "task_net_in_byte", "task_net_out_byte"]

    for trial_dir in os.listdir(os.path.join(root_dir, "pai")):
        summary_dir = os.path.join(root_dir, "summary", trial_dir)
        os.makedirs(summary_dir, exist_ok=True)
        writer = SummaryWriter(summary_dir)
        trial_dir = os.path.join(root_dir, "pai", trial_dir)
        for category in categories:
            with open(os.path.join(trial_dir, category + ".json"), "r") as fp:
                data = json.load(fp)
            data = data["data"]["result"][0]["values"]
            if not data:
                continue
            start_time = data[0][0]
            for j, (time, val) in enumerate(data, start=1):
                writer.add_scalar(category, float(val), time - start_time)
        writer.close()


def average(iterable):
    ss, cnt = None, 0
    for x in iterable:
        ss = x if ss is None else ss + x
        cnt += 1
    return ss / cnt


def sum_durations(durations):
    ret = timedelta(seconds=0)
    for d in durations:
        ret += d
    return ret


class EpochMetrics(object):
    def __init__(self):
        self.sampled_steps = []
        self.data_time_samples = []
        self.batch_time_samples = []
        self.start_time = datetime.now()
        self.finish_time = datetime.now()
        self.total_steps = 0

    def start(self, time):
        self.start_time = time

    def update(self, time, step, total_steps, data_time, batch_time):
        self.total_steps = total_steps
        self.sampled_steps.append(step)
        self.data_time_samples.append(data_time)
        self.batch_time_samples.append(batch_time)

    def finish(self, time):
        self.finish_time = time

    @property
    def duration(self):
        return self.finish_time - self.start_time

    def __str__(self):
        return "{} ({} - {})".format(self.duration, self.start_time, self.finish_time)


class EvaluationMetrics(object):
    def __init__(self):
        self.elapsed_time = []
        self.finish_time = []
        self.latest_start_time = None

    def update(self, time):
        if self.latest_start_time is None:
            self.latest_start_time = time

    def finish(self, time):
        self.elapsed_time.append(time - self.latest_start_time)
        self.finish_time.append(time)
        self.latest_start_time = None

    def __str__(self):
        return "Evaluations ({} total)".format(len(self.elapsed_time))


# processing states
INITIALIZED = 0
DOWNLOADING = 1
LAUNCHED = 2
TRAINING = 3
EVALUATING = 4
FINISHED = 5


class TrainingLogAnalyzer(object):
    def process_line(self, state, line):
        match_time = [r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d+)\]",
                      r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d+)"]
        timestamp = None
        for time in match_time:
            time_search_result = re.search(time, line)
            if time_search_result is None or time_search_result.pos != 0:
                # skip line
                continue
            line = line[len(time_search_result.group(0)):].lower().strip()
            timestamp = dateutil_parser.parse(time_search_result.group(1))
            break

        if not timestamp:
            return state

        timestamp += timedelta(hours=8)  # time zone correction

        if "successfully copied hdfs file" in line:
            if state != DOWNLOADING:
                self.download_start_time = timestamp
            return DOWNLOADING
        elif line == "process launched":
            if state != LAUNCHED:
                self.process_launch_time = timestamp
            return LAUNCHED
        elif re.match(r"epoch[: \[]*\d+", line):
            epoch_num = int(re.search(r"epoch[: \[]*(\d+)", line).group(1))
            while epoch_num >= len(self.training_metrics):
                self.training_metrics.append(EpochMetrics())
            format = r"epoch: \[\d+\]\[\s*(\d+)/(\d+)\]\s+time\s+(.*?)\s+\(.*?\)\s+data\s+(.*?)\s+\(.*?\)"
            matching_results = re.search(format, line)
            if matching_results is None:
                if "start" in line:
                    self.training_metrics[epoch_num].start(timestamp)
            else:
                step = int(matching_results.group(1))
                total_step = int(matching_results.group(2))
                batch_time = timedelta(seconds=float(matching_results.group(3)))
                data_time = timedelta(seconds=float(matching_results.group(4)))
                self.training_metrics[epoch_num].update(timestamp, step, total_step, data_time, batch_time)
            if state == EVALUATING:
                self.evaluation_metrics.finish(timestamp)
            if state != TRAINING:
                self.training_metrics[epoch_num].start(timestamp)
            return TRAINING
        elif re.match(r"evaluation", line):
            if state == TRAINING:
                self.training_metrics[-1].finish(timestamp)
            self.evaluation_metrics.update(timestamp)
            return EVALUATING
        elif re.search(r"subprocess terminated", line):
            if state == TRAINING:
                self.training_metrics[-1].finish(timestamp)
            if state == EVALUATING:
                self.evaluation_metrics.finish(timestamp)
            self.stop_time = timestamp
            return FINISHED
        else:
            # print("=> Skipping log:", line)
            return state

    def analyze_training_logs(self, root_dir, workbook):
        date_format = workbook.add_format({"num_format": "hh:mm:ss.000"})

        log_summary = workbook.add_worksheet("log_summary")
        row_number = 1

        first_job_launch_time = 10 ** 18
        for i, trial_name in enumerate(os.listdir(os.path.join(root_dir, "pai")), start=1):
            trial_dir = os.path.join(root_dir, "pai", trial_name)
            with open(os.path.join(trial_dir, "summary.json"), "r") as fp:
                summaries = json.load(fp)
            first_job_launch_time = min(first_job_launch_time, summaries["jobStatus"]["createdTime"] // 1000)
        first_job_launch_time = datetime.fromtimestamp(first_job_launch_time)

        for i, trial_name in enumerate(os.listdir(os.path.join(root_dir, "pai")), start=1):
            trial_dir = os.path.join(root_dir, "pai", trial_name)
            with open(os.path.join(trial_dir, "summary.json"), "r") as fp:
                summaries = json.load(fp)

            create_time = summaries["jobStatus"]["createdTime"] // 1000
            app_launch_time = summaries["jobStatus"]["appLaunchedTime"] // 1000
            create_time = datetime.fromtimestamp(create_time)
            app_launch_time = datetime.fromtimestamp(app_launch_time)

            for container_name in os.listdir(trial_dir):
                container_dir = os.path.join(trial_dir, container_name)
                if not os.path.isdir(container_dir):
                    continue
                print("=> Processing %s: %s" % (trial_name, container_name))
                log_path = os.path.join(container_dir, "user.pai.stdout")
                state = INITIALIZED

                self.download_start_time = datetime.now()
                self.process_launch_time = datetime.now()
                self.stop_time = datetime.now()
                self.training_metrics = []
                self.evaluation_metrics = EvaluationMetrics()

                with open(log_path, "r") as f:
                    for line in f.readlines():
                        state = self.process_line(state, line)
                if state == EVALUATING:
                    self.evaluation_metrics.finish(datetime.now())

                # interested metrics
                interested_metrics = {
                    "job_delay": (create_time - first_job_launch_time),
                    "job_wait_time": (app_launch_time - create_time),
                    "preparation_time": (self.download_start_time - app_launch_time),
                    "hdfs_download_time": (self.process_launch_time - self.download_start_time),
                    "time_till_first_epoch": (self.training_metrics[0].start_time - create_time),
                    "first_step_data_time": (self.training_metrics[0].data_time_samples[0]),
                    "first_step_tot_time": (self.training_metrics[0].batch_time_samples[0]),
                    "first_epoch_data_time_per_step": average(self.training_metrics[0].data_time_samples),
                    "first_epoch_tot_time_per_step": average(self.training_metrics[0].batch_time_samples),
                    "first_epoch_train_time": self.training_metrics[0].duration,
                    "first_epoch_evaluation_time": self.evaluation_metrics.elapsed_time[0],
                    "train_data_time_per_step": average([average(t.data_time_samples) for t in self.training_metrics]),
                    "train_time_per_step": average([average(t.batch_time_samples) for t in self.training_metrics]),
                    "total_epochs": len(self.training_metrics),
                    "total_training_time": sum_durations([t.duration for t in self.training_metrics]),
                    "total_eval_time": sum_durations(self.evaluation_metrics.elapsed_time),
                    "time_till_stop": self.stop_time - self.evaluation_metrics.finish_time[-1]
                }

                log_summary.write(0, 0, "trial")
                log_summary.write(0, 1, "container")
                log_summary.write(row_number, 0, trial_name)
                log_summary.write(row_number, 1, container_name)
                for od, k in enumerate(sorted(interested_metrics.keys()), start=2):
                    log_summary.write(0, od, k)
                    val = interested_metrics[k]
                    if isinstance(val, int) or isinstance(val, float):
                        log_summary.write_number(row_number, od, val)
                    else:
                        # timedelta
                        log_summary.write_datetime(row_number, od, val, date_format)

                row_number += 1


def main(root_dir):
    workbook = xlsxwriter.Workbook(os.path.join(root_dir, 'result.xlsx'))
    analyze_metrics(root_dir)
    # metrics_analyzer = TrainingLogAnalyzer()
    # metrics_analyzer.analyze_training_logs(root_dir, workbook)
    workbook.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="Where the downloader stores the data in the last step")
    args = parser.parse_args()
    main(args.dir)
