#!/usr/bin/env python

import datetime
import pprint
import os
import shutil
import subprocess
import sysconfig
import xmltodict
import yaml


version = "1.3.0"


def get_cpu_info():
    proc_cpuinfo_file = "/proc/cpuinfo"

    if os.path.exists(proc_cpuinfo_file):
        cpu_info = []
        core_info = {}

        # Read proc_cpuinfo_file
        proc_cpuinfo_object = open(proc_cpuinfo_file, "r")

        for line in proc_cpuinfo_object.readlines():
            # Skip blank lines
            if line.strip() != "":
                # Covert line to list on :
                proc_cpuinfo_line_list = [x.strip() for x in line.split(":")]
                # If first element is processor start new cpu list
                if proc_cpuinfo_line_list[0] == "processor" and core_info != {}:
                    cpu_info.append(core_info)
                    core_info = {}

                # Add element to the dict
                core_info[proc_cpuinfo_line_list[0]] = proc_cpuinfo_line_list[1]

        cpu_info.append(core_info)
        proc_cpuinfo_object.close()

        return cpu_info
    else:
        return None


def get_gpu_info():
    if shutil.which("nvidia-smi") != None:
        # Run nvidia-smi -q -x ; return None if the command exits with an error
        try:
            nvidia_smi_xml = subprocess.check_output(["nvidia-smi", "-q", "-x"])
        except subprocess.CalledProcessError as e:
            return None

        # Convert the XML output from nvidia-smi to a dict and return
        return xmltodict.parse(nvidia_smi_xml)
    else:
        return None


def get_os_info():
    os_release_file = "/etc/os-release"

    if os.path.exists(os_release_file):
        os_info = {}

        # Read os_release_file
        os_release_object = open(os_release_file, "r")

        for line in os_release_object.readlines():
            # Skip blank lines
            if line.strip() != "":
                # Covert line to list on =
                os_release_line_list = [x.strip() for x in line.split("=")]

                # Add element to the dict
                os_info[os_release_line_list[0]] = os_release_line_list[1]

        os_release_object.close()

        return os_info
    else:
        return None


def get_dmi_info():
    dmi_info = {}

    dmiinfo_files = [
        "/sys/class/dmi/id/product_family",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/board_vendor",
    ]

    for dmiinfo_file in dmiinfo_files:
        if os.path.exists(dmiinfo_file):
            dmiinfo_object = open(dmiinfo_file, "r")
            os.path.basename(os.path.normpath(dmiinfo_file))

            dmi_info[
                os.path.basename(os.path.normpath(dmiinfo_file))
            ] = dmiinfo_object.readline().strip()
            dmiinfo_object.close()

    if dmi_info == {}:
        return None
    else:
        return dmi_info


def get_environ_info():
    environ_info = {}

    for key, value in os.environ.items():
        environ_info[key] = value

    return environ_info


def get_system_info():
    system_info = {}

    system_info["datetime"] = str(datetime.datetime.now())
    system_info["version"] = version
    system_info["hostname"] = os.uname()[1]
    system_info["cpu"] = get_cpu_info()
    system_info["gpu"] = get_gpu_info()
    system_info["dmi"] = get_dmi_info()
    system_info["os"] = get_os_info()
    system_info["python"] = sysconfig.get_config_vars()
    system_info["env"] = get_environ_info()

    return system_info


if __name__ == "__main__":
    sysinfo = get_system_info()

    with open("system_info.yaml", "w") as system_info_file:
        yaml.dump(sysinfo, system_info_file, default_flow_style=False)
