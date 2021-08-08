import subprocess
from subprocess import PIPE


def find_filename(url):

    proc = subprocess.Popen(
        ["java", "-jar", "/home/dev3/SaeheeJeon/cut-parser-1.1-all.jar", url],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    # proc = subprocess.call(
    #         ['java','-jar', '/home/dev3/SaeheeJeon/cut-parser-1.0-all.jar', '553038.hwp'],
    #         stdin=PIPE, stdout=PIPE, stderr=PIPE)

    output = proc.communicate()[
        0
    ]  ## this will capture the output of script called in the parent script.

    txt = output.decode("utf-8")

    for t in txt.split(" "):

        if t.endswith(".png") or t.endswith(".jpg") or t.endswith(".jpeg"):
            return t

    return ""
