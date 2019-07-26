import json
import os
from argparse import ArgumentParser
from datetime import timedelta

import xlsxwriter


def analyze_metrics(root_dir, worksheet):
    columns = ["task_block_in_byte", "task_block_out_byte", "task_cpu_percent",
               "task_gpu_mem_percent", "task_gpu_percent", "task_mem_usage_byte",
               "task_net_in_byte", "task_net_out_byte"]

    for i, column in enumerate(columns, start=1):
        worksheet.write(0, i, column)

    for trial_dir in os.listdir(os.path.join(root_dir, "pai")):
        trial_dir = os.path.join(root_dir, "pai", trial_dir)
        for i, column in enumerate(columns, start=1):
            with open(os.path.join(trial_dir, column + ".json"), "r") as fp:
                data = json.load(fp)
            data = data["data"]["result"][0]["values"]
            if not data:
                continue
            start_time = data[0][0]
            for j, (time, val) in enumerate(data, start=1):
                worksheet.write_datetime(j, 0, timedelta(seconds=time - start_time))
                worksheet.write_number(j, i, val)



def main(root_dir):
    workbook = xlsxwriter.Workbook(os.path.join(root_dir, 'result.xlsx'))
    metrics = workbook.add_worksheet("metrics")
    analyze_metrics(root_dir, metrics)
    workbook.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="Where the downloader stores the data in the last step")
    args = parser.parse_args()
    main(args.dir)
