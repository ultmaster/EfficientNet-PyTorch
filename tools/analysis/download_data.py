import json
import os
import time
from argparse import ArgumentParser

import requests
from bs4 import BeautifulSoup


def write_json(fname, data):
    with open(fname, "w") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)


def main(nni_ip, pai_ip):
    experiment_url = "http://%s/api/v1/nni/experiment" % nni_ip
    trial_jobs_url = "http://%s/api/v1/nni/trial-jobs" % nni_ip
    metric_data_url = "http://%s/api/v1/nni/metric-data" % nni_ip
    experiment_data = requests.get(experiment_url).json()
    trial_jobs_data = requests.get(trial_jobs_url).json()
    metric_data = requests.get(metric_data_url).json()

    experiment_id = experiment_data["id"]

    output_dir = os.path.join("tmp", "analysis", nni_ip.replace(":", "."), experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    print("=> Getting NNI ready...")
    write_json(os.path.join(output_dir, "experiment.json"), experiment_data)
    write_json(os.path.join(output_dir, "trial_jobs.json"), trial_jobs_data)
    write_json(os.path.join(output_dir, "metric_data.json"), metric_data)

    for i, job in enumerate(trial_jobs_data):
        job_id = job["id"]
        print("=> Getting job %s [%d/%d]" % (job_id, i + 1, len(trial_jobs_data)))
        job_summary_pai_url = "http://%s/rest-server/api/v2/user/v_yugzh/jobs/nni_exp_%s_trial_%s" % (
            pai_ip, experiment_id, job_id
        )
        job_summary = requests.get(job_summary_pai_url).json()
        job_name = job_summary["name"]  # job name on pai
        job_username = job_summary["jobStatus"]["username"]
        job_output_dir = os.path.join(output_dir, "pai", job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        write_json(os.path.join(job_output_dir, "summary.json"), job_summary)
        fetched_status_id = set()
        for task_roles in job_summary["taskRoles"].values():
            for status in task_roles["taskStatuses"]:
                status_id = status["containerIp"] + "_" + status["containerId"]
                status_url = status["containerLog"]
                if status_id in fetched_status_id:
                    print("=> There is a conflict in %s, conflicted status is %s" % (job_id, status_id))
                fetched_status_id.add(status_id)
                os.makedirs(os.path.join(job_output_dir, status_id), exist_ok=True)

                # Getting logs in container
                for file in ["stdout", "stderr", "user.pai.stdout", "user.pai.stderr",
                             "runtime.docker.pai.error", "runtime.docker.pai.log", "runtime.pai.agg.error",
                             "runtime.yarn.pai.log", "runtime.yarn.pai.error"]:
                    file_url = status_url + "/" + file + "/?start=0"
                    file_html = requests.get(file_url).text
                    file_path = os.path.join(job_output_dir, status_id, file)
                    soup = BeautifulSoup(file_html, 'html.parser')
                    with open(file_path, "w") as fp:
                        for k, pre in enumerate(soup.find_all("pre")):
                            if k > 0:
                                fp.write("\n\n===========================\n\n")
                            else:
                                print("=> Outputing contents to %s" % file_path)
                            fp.write(pre.get_text())

                # Getting metrics
                metrics = ["task_cpu_percent{%s}", "task_mem_usage_byte{%s}", "task_net_in_byte{%s}",
                           "task_net_out_byte{%s}", "irate(task_block_in_byte{%s}[300s])",
                           "irate(task_block_out_byte{%s}[300s])",
                           "task_gpu_percent{%s}", "task_gpu_mem_percent{%s}"]
                start_time = job_summary["jobStatus"]["createdTime"] / 1000
                if job_summary["jobStatus"]["state"] == "RUNNING":
                    end_time = time.time()
                else:
                    end_time = job_summary["jobStatus"]["completedTime"] / 1000
                metrics_query_url = "http://%s/prometheus/api/v1/query_range" % pai_ip
                for metric in metrics:
                    query_params = {
                        "query": 'avg by (job_name)(%s)' % (
                                metric % ("job_name=~\"%s~%s\"" % (job_username, job_name))),
                        "start": int(start_time),
                        "end": int(end_time),
                        "step": 10
                    }
                    data = requests.get(metrics_query_url, params=query_params).json()
                    alias = metric[:metric.index("{")]
                    if "(" in alias:
                        alias = alias[alias.index("(") + 1:]
                    metric_path = os.path.join(job_output_dir, alias + ".json")
                    write_json(metric_path, data)
                    print("=> Writing metrics to %s" % metric_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("nni_ip", help="Please provide the IP address of your NNI portal, "
                                       "from which we gather data. In the format of xxx.xxx.xxx.xxx:xxxx")
    parser.add_argument("pai_ip", help="Please provide the IP address of PAI, in the format of "
                                       "xxx.xxx.xxx.xxx")
    args = parser.parse_args()

    main(args.nni_ip, args.pai_ip)
