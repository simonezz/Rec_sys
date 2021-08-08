import sys

# sys.path.append('hml_equation_parser')
import subprocess
from subprocess import PIPE
from hml_equation_parser.hulkEqParser import hmlEquation2latex
import time




"""
hwplib 자바 파일들을 이용해서 hwp파싱
"""

#
# url = " https://s3.ap-northeast-2.amazonaws.com/mathflat/math_problems/hwp/Mo/MO_200711/h3/200711_Ha_B/3_p.hwp"
# url = " https://s3.ap-northeast-2.amazonaws.com/mathflat/math_problems/hwp/Mo/MO_201810/h3/201810_Se_A/15_p.hwp"
url = "https://s3.ap-northeast-2.amazonaws.com/mathflat/math_problems/hwp/9/e/1/1/01900/9_11101900_WGg7F_uRp_p.hwp"


'''
hwp url로부터 한글을 파싱하는 함수 hwp -> hwp text -> latex text
Input : hwp url
Output : plainText + latex
'''
def hwp_parser(url):
    sys.path.append("./hml_equation_parser")
    # try:
    # print(url)
    proc = subprocess.Popen(
        # ['java', '-jar', '/home/ubuntu/Recommender_SH/utils/hwp.jar', url],
        ["java", "-jar", "/home/dev3/SaeheeJeon/Rec_sys/utils/hwp.jar", url],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )

    output = proc.communicate()[
        0
    ]  ## this will capture the output of script called in the parent script.

    txt = output.decode("utf-8")

    # print("raw_text from hwp: ", txt)
    # except: return False
    result_txt = []

    # try:
    for t in list(txt.split("\n"))[1:]:
        if len(t) > 0:
            result_txt.append(hmlEquation2latex(t))
    # except TimeOutException as e:
    #     print(e)
    #     return False

    return " ".join(result_txt)


if __name__ == "__main__":
    print("after parsing: ", hwp_parser(url))
