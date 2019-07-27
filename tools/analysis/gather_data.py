import json
import os
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import timedelta

import xlsxwriter


def analyze_metrics(root_dir, workbook):
    date_format = workbook.add_format({"num_format": "hh:mm:ss"})
    categories = ["task_block_in_byte", "task_block_out_byte", "task_cpu_percent",
                  "task_gpu_mem_percent", "task_gpu_percent", "task_mem_usage_byte",
                  "task_net_in_byte", "task_net_out_byte"]
    for category in categories:
        worksheet = workbook.add_worksheet(category)
        for i, trial_dir in enumerate(os.listdir(os.path.join(root_dir, "pai")), start=1):
            worksheet.write(0, i, trial_dir)
            trial_dir = os.path.join(root_dir, "pai", trial_dir)
            with open(os.path.join(trial_dir, category + ".json"), "r") as fp:
                data = json.load(fp)
            data = data["data"]["result"][0]["values"]
            if not data:
                continue
            start_time = data[0][0]
            for j, (time, val) in enumerate(data, start=1):
                worksheet.write_datetime(j, 0, timedelta(seconds=time - start_time), date_format)
                worksheet.write_number(j, i, float(val))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, fmt=':f'):
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def analyze_training_logs(root_dir, workbook):
    for i, trial_name in enumerate(os.listdir(os.path.join(root_dir, "pai")), start=1):
        trial_dir = os.path.join(root_dir, "pai", trial_name)
        with open(os.path.join(root_dir, "summary.json"), "r") as fp:
            summaries = json.load(fp)
        create_time = summaries["jobStatus"]["createdTime"]
        app_launch_time = summaries["jobStatus"]["appLaunchedTime"]
        for container_name in os.listdir(trial_dir):
            container_dir = os.path.join(trial_dir, container_name)
            log_path = os.path.join("user.pai.stdout")

            # interested metrics
            launch_before_creation_time = (app_launch_time - create_time) / 1000
            create_before_download_time = AverageMeter()
            hdfs_download_time = AverageMeter()
            time_till_first_epoch = AverageMeter()
            first_step_data_time = AverageMeter()
            first_step_tot_time = AverageMeter()
            first_epoch_data_time_per_step = AverageMeter()
            first_epoch_tot_time_per_step = AverageMeter()
            first_epoch_eval_time_per_step = AverageMeter()
            first_epoch_train_time = AverageMeter()
            first_epoch_eval_time = AverageMeter()
            train_data_time_per_step = AverageMeter()
            train_tot_time_per_step = AverageMeter()
            train_time_per_step = AverageMeter()
            eval_time_per_step = AverageMeter()
            total_epochs = AverageMeter()
            total_train_time = AverageMeter()
            total_eval_time = AverageMeter()
            total_time = AverageMeter()
            hdfs_upload_time = AverageMeter()
            time_after_last_epoch = AverageMeter()

            with open(log_path, "r") as f:
                for line in f.readlines():
                    line = line.strip().lower()


        worksheet.write(0, i, trial_dir)
        trial_dir = os.path.join(root_dir, "pai", trial_dir)
        with open(os.path.join(trial_dir, category + ".json"), "r") as fp:
            data = json.load(fp)
        data = data["data"]["result"][0]["values"]
        if not data:
            continue
        start_time = data[0][0]
        for j, (time, val) in enumerate(data, start=1):
            worksheet.write_datetime(j, 0, timedelta(seconds=time - start_time), date_format)
            worksheet.write_number(j, i, float(val))



def main(root_dir):
    workbook = xlsxwriter.Workbook(os.path.join(root_dir, 'result.xlsx'))
    analyze_metrics(root_dir, workbook)
    workbook.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="Where the downloader stores the data in the last step")
    args = parser.parse_args()
    main(args.dir)
