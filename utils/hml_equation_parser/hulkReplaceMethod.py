# -*- coding: utf-8 -*-
import sys
import re

import codecs
import json
import os
from typing import Dict, Tuple

with codecs.open(
    os.path.join(os.path.dirname(__file__), "convertMap.json"), "r", "utf8"
) as f:
    convertMap = json.load(f)

barDict = convertMap["BarConvertMap"]
matDict = convertMap["MatrixConvertMap"]
braceDict = convertMap["BraceConvertMap"]


def isHangul(text):
    # Check the Python Version
    pyVer3 = sys.version_info >= (3, 0)

    if pyVer3:  # for Ver 3 or later
        encText = text
    else:  # for Ver 2.x
        if type(text) is not unicode:
            encText = text.decode("utf-8")
        else:
            encText = text

    hanCount = len(re.findall(u"[\u3130-\u318F\uAC00-\uD7A3]+", encText))
    return hanCount > 0


def _findOutterBrackets(eqString: str, startIdx: int) -> Tuple[int, int]:
    """
    eqString : equation string for converting.
    startIdx : the cursor of equation string to find brackets.
    return:
        (startCursor, endCursor) for outter brackets.
    """
    idx = startIdx
    while True:
        idx -= 1
        if eqString[idx] == "{":
            break

    return _findBrackets(eqString, idx, direction=1)


def _findBrackets(eqString: str, startIdx: int, direction: int) -> Tuple[int, int]:
    """
    eqString : equation string for converting.
    startIdx : the cursor of equation string to find brackets.
    direction : the direction of find.
        if 0, find brackets before the cursor.
           1, find brackets after the cursor.
    return:
        (startCursor, endCursor) for brackets.
    """
    if direction == 1:
        startCur = eqString.find(r"{", startIdx)
        bracketCount = 1
        for i in range(startCur + 1, len(eqString)):
            if eqString[i] == r"{":
                bracketCount += 1
            elif eqString[i] == r"}":
                bracketCount -= 1

            if bracketCount == 0:
                return (startCur, i + 1)
    else:
        # reverse string and convert brackets.
        eqString = eqString[::-1]
        for idx, char in enumerate(eqString):
            if char == r"{":
                eqString = eqString[0:idx] + r"}" + eqString[idx + 1 :]
            if char == r"}":
                eqString = eqString[0:idx] + r"{" + eqString[idx + 1 :]

        # find brackets with new cursor
        newStartIdx = len(eqString) - (startIdx + 1)
        startCur, endCur = _findBrackets(eqString, newStartIdx, direction=1)
        return (len(eqString) - endCur, len(eqString) - startCur)

    raise ValueError("cannot find bracket")


def replaceAllBar(eqString: str) -> str:
    """
    replace hat-like equation string.
    """

    def replaceBar(eqString: str, barStr: str, barElem: str) -> str:
        cursor = 0

        while True:
            cursor = eqString.find(barStr)
            if cursor == -1:
                break
            try:
                eStart, eEnd = _findBrackets(eqString, cursor, direction=1)
                if eStart == -1:  # ?????? bracket ?????? ??????
                    elem = (
                        "{" + list(eqString[cursor + len(barStr) :].split(" "))[0] + "}"
                    )
                    beforeBar = eqString[0:cursor]
                    afterBar = eqString[cursor + len(barStr) + len(elem) - 2 :]
                else:
                    bStart, bEnd = _findOutterBrackets(eqString, cursor)
                    elem = eqString[eStart:eEnd]

                    beforeBar = eqString[0:cursor]
                    afterBar = eqString[eEnd:]

                eqString = beforeBar + barElem + elem + afterBar
            except ValueError:
                return eqString
        return eqString

    for barKey, barElem in barDict.items():
        eqString = replaceBar(eqString, barKey, barElem)
    return eqString


def replaceAllMatrix(eqString: str) -> str:
    """
    replace matrix-like equation
    """

    def replaceElementsOfMatrix(bracketStr: str) -> str:
        """
        replace the elements of matrix
        """
        bracketStr = bracketStr[1:-1]  ## remove brackets

        bracketStr = bracketStr.replace(r"#", r" \\ ")
        bracketStr = bracketStr.replace(r"&amp;", r"&")

        return bracketStr

    def replaceMatrix(eqString: str, matStr: str, matElem: Dict[str, object]) -> str:

        while True:
            cursor = eqString.find(matStr)
            if cursor == -1:
                break
            try:
                eStart, eEnd, eqString = _findBrackets2(
                    eqString, cursor + len(matStr) - 1, direction=1
                )
                elem = replaceElementsOfMatrix(eqString[eStart : eEnd + 1])

                if matElem["removeOutterBrackets"] == True:
                    # try:
                    #     # bStart, bEnd = _findOutterBrackets(eqString, cursor+len(matStr)-1)
                    #     # if bStart<cursor:
                    #     #     beforeMat = eqString[0:bStart]
                    #     # else:
                    #     #     beforeMat = eqString[0:bStart - len(matStr) - 1]
                    #     # afterMat = eqString[bEnd:]
                    #
                    #     beforeMat = eqString[0:cursor]
                    #     afterMat = eqString[eEnd+1:]
                    # except:
                    beforeMat = eqString[0:cursor]

                    if eEnd:

                        afterMat = eqString[eEnd + 1 :]
                    else:
                        afterMat = " ".join(
                            (eqString[cursor + len(matStr) :].split(" "))[1:]
                        )
                else:
                    beforeMat = eqString[0:cursor]
                    afterMat = eqString[eEnd + 1 :]

                eqString = (
                    beforeMat[:cursor]
                    + matElem["begin"]
                    + elem
                    + matElem["end"]
                    + afterMat
                )
            except ValueError:
                return eqString
        return eqString

    for matKey, matElem in matDict.items():
        eqString = replaceMatrix(eqString, matKey, matElem)
    return eqString


def replaceRootOf(eqString: str) -> str:
    """
    `root {1} of {2}` -> `\sqrt[1]{2}`
    'root
    """
    rootStr = r"root"
    ofStr = r"of"

    while True:
        rootCursor = eqString.find(rootStr)
        if rootCursor == -1:
            break
        try:
            ofCursor = eqString.find(ofStr)

            elem1 = _findBrackets(eqString, rootCursor, direction=1)
            elem2 = _findBrackets(eqString, ofCursor, direction=1)

            e1 = eqString[elem1[0] + 1 : elem1[1] - 1]
            e2 = eqString[elem2[0] + 1 : elem2[1] - 1]

            eqString = (
                eqString[0:rootCursor]
                + r"\sqrt"
                + r"["
                + e1
                + r"]"
                + r"{"
                + e2
                + r"}"
                + eqString[elem2[1] + 1 :]
            )
        except ValueError:
            return eqString
    return eqString


def _findBrackets2(
    eqString, cursor, direction=0
):  # eqString?????? cursor ????????? '{'??? '}'??? ?????? ??????

    # direction = 0?????? cursor ????????? bracket ?????? 1?????? cursor ????????? ??????

    if direction == 1:
        i = cursor + 1
        if "{" not in eqString[i:]:
            return False, False, eqString
        i = eqString[cursor + 1 :].index("{") + cursor + 1  # "{"??? ?????? ??????

        if (
            eqString[cursor + 1 : i].strip() != ""
        ):  # root??? ?????? ????????? {????????? ?????? ????????? ?????? sqrt ?????? ?????????(?????? ????????? ????????? ?????? ??????) ex) 3over5 {~~~
            return False, False, eqString

        else:
            startCur = i

            i = startCur + 1

            tmp = 1
            while True:
                if i >= len(eqString):
                    eqString = eqString + "}"
                    break
                if tmp == 0:
                    break
                if eqString[i] == "}":
                    tmp -= 1
                elif eqString[i] == "{":
                    tmp += 1
                else:
                    if isHangul(eqString[i]):  # ?????? ????????? ????????? ??? ?????? ???????????? ?????? ??????
                        eqString = eqString[:i] + "}" + eqString[i:]
                        i += 1
                        break
                    pass
                i += 1

            endCur = i - 1

    else:  # direction=0 ??? ???
        i = cursor - 1
        if "{" not in eqString[: i + 1]:
            return False, False, eqString

        while True:
            if i == 0:
                return False, False, eqString
            if eqString[i] == "}":
                break
            i -= 1

        endCur = i

        tmp = 1

        if (
            eqString[i + 1 : cursor].strip() != ""
        ):  # "}"??? root ????????? ?????? ????????? ?????? sqrt ?????? ?????????(?????? ????????? ????????? ?????? ??????) ex) 1over4
            return False, False, eqString
        else:
            i = endCur - 1

            while True:
                if i < 0:
                    eqString = "{" + eqString
                    i = 0
                    break

                if tmp == 0:
                    break
                if eqString[i] == "{":
                    tmp -= 1
                elif eqString[i] == "}":
                    tmp += 1
                else:
                    pass
                i -= 1
            startCur = i + 1
    return startCur, endCur, eqString


def replaceRootOf2(eqString: str) -> str:
    """
    `root {1} of {2}` -> `\sqrt[1]{2}`
    'root {3} -> \sqrt{3}

    """

    hmlFracString = r"root"
    latexFracString = r"\sqrt"
    ofString = r"of"

    while True:
        cursor = eqString.find(hmlFracString)  # "over"?????? ??????

        if cursor == -1:  # ????????? root ????????? ???????????? ??????
            break

        ofCursor = eqString.find(ofString)

        # of ?????? ???
        if (
            ofCursor == -1
            or ofCursor < cursor
            or hmlFracString in eqString[cursor:ofCursor]
        ):

            start, end, eqString = _findBrackets2(
                eqString, cursor + 3, direction=1
            )  # of ?????? ?????? ??????

            if start:  # root ?????? brackets?????? ???????????? ???
                afterRoot = eqString[start : end + 1]
                i = 0

                eqString = (
                    eqString[0:cursor]
                    + latexFracString
                    + "{"
                    + afterRoot
                    + "}"
                    + eqString[end + 1 :]
                )

            else:  # root3??? ?????? bracket ??????

                strList = eqString[cursor + 4 :].split(" ")

                i = 0
                while True:
                    if strList[i] != " ":
                        afterRoot = strList[i]
                        break
                    i += 1

                    #######

                eqString = (
                    eqString[0:cursor]
                    + latexFracString
                    + "{"
                    + afterRoot
                    + "}"
                    + " ".join(strList[i + 1 :])
                )

        else:  # of ?????? ???
            start, end, eqString = _findBrackets2(
                eqString, ofCursor + 1, direction=1
            )  # of ?????? ?????? ??????

            if start:  # of ?????? brackets?????? ???????????? ???
                afterOf = eqString[start : end + 1]
                i = 0

            else:
                strList = eqString[ofCursor:].split(" ")
                if strList[0] != ofString:  # of2 ??? ?????? ??????????????? ???????????? ?????? ??????
                    afterOf = strList[0][2:]
                else:
                    i = 1
                    while True:
                        if strList[i] != " ":
                            afterOf = strList[i]
                            break
                        i += 1

            eqString = (
                eqString[0:cursor]
                + r"\sqrt"
                + r"["
                + eqString[cursor + 4 : ofCursor]
                + r"]"
                + r"{"
                + afterOf
                + r"}"
                + " ".join(strList[i + 1 :])
            )
    return eqString


def replaceFrac(eqString: str) -> str:
    """
    `{1} over {2}` -> `\frac{1}{2}`
    """
    hmlFracString = r"over"
    latexFracString = r"\frac"

    while True:
        cursor = eqString.find(hmlFracString)

        if cursor == -1:
            break
        try:
            # find numerator
            numStart, numEnd = _findBrackets(eqString, cursor, direction=0)
            numerator = eqString[numStart:numEnd]

            beforeFrac = eqString[0:numStart]
            afterFrac = eqString[cursor + len(hmlFracString) :]

            eqString = beforeFrac + latexFracString + numerator + afterFrac
        except ValueError:
            return eqString
    return eqString


def replaceFrac_no_bracket(String):  # 3 over 4 -> \frac{3}{4}
    hmlFracString = r"over"
    latexFracString = r"\frac"

    strList = String.split(" ")
    cursor = strList.index(hmlFracString)
    numerator = "{" + strList[cursor - 1] + "}"
    denominator = "{" + strList[cursor + 1] + "}"

    return [latexFracString, numerator, denominator]


def replaceFrac2(eqString: str) -> str:
    """
    `{1} over {2}` -> `\frac{1}{2}`
    """
    hmlFracString = r"over"
    latexFracString = r"\frac"

    while True:
        cursor = eqString.find(hmlFracString)  # "over"?????? ??????

        if cursor == -1:
            break

        # find numerator
        numStart, numEnd, eqString = _findBrackets2(eqString, cursor, direction=0)
        # numStart??? bracket { ??????, numEnd??? bracket }??? ?????? +1

        strList = eqString[:cursor].split(" ")
        i = len(strList) - 1
        if (
            numStart and eqString[numEnd + 1 : cursor].strip() != ""
        ):  # "}"??? over????????? ???????????? ??????
            numerator = eqString[numEnd + 1 : cursor].strip()
            beforeFrac = eqString[: numEnd + 1]

        elif numStart:
            numerator = eqString[numStart : numEnd + 1]
            beforeFrac = eqString[:numStart]

        elif eqString[cursor - 1] != " ":  # 4over3??? ?????? ????????? over ?????? ????????????
            strList = eqString[:cursor].split(" ")
            numerator = strList[-1].strip()  # ??????
            beforeFrac = " ".join(strList[: len(strList) - 1])

        else:  # 4 over 3 ??? ?????? ??????

            i = len(strList) - 1

            while True:

                if strList[i].strip() != "":
                    numerator = strList[i]
                    break
                i -= 1
            beforeFrac = " ".join(strList[:i])

        # find denominator
        numStart, numEnd, eqString = _findBrackets2(eqString, cursor + 3, direction=1)
        # numStart??? bracket { ??????, numEnd??? bracket }??? ?????? +1
        j = 0

        if (
            numStart and eqString[cursor + 4 : numStart].strip() != ""
        ):  # over??? "{"????????? ???????????? ??????

            denominator = eqString[cursor + 4 : numStart].strip()
            deStrList = eqString[numStart:].split(" ")

        elif numStart:

            denominator = eqString[numStart : numEnd + 1]
            deStrList = eqString[numEnd + 1 :].split(" ")

        elif eqString[cursor + 1] != " ":  # 4over3??? ?????? ????????? over ?????? ????????????

            deStrList = eqString[cursor + 4 :].split(" ")
            denominator = deStrList[0].strip()  # ??????

        else:  # 4 over 3 ??? ?????? ??????

            deStrList = eqString[cursor + 4 :].split(" ")
            j = 0

            while True:

                if deStrList[j] != " ":
                    denominator = deStrList[j]
                    break

                i += 1

        eqString = (
            beforeFrac
            + latexFracString
            + "{"
            + numerator
            + "}"
            + "{"
            + denominator
            + "}"
            + " ".join(deStrList[j + 1 :])
        )
    # 2over3 -> \frac{2}{3}

    return eqString


def replaceAllBrace(eqString: str) -> str:
    """
    replace (over, under)brace equation string.
    """

    def replaceBrace(eqString: str, braceStr: str, braceElem: str) -> str:
        cursor = 0

        while True:
            cursor = eqString.find(braceStr)
            if cursor == -1:
                break
            try:
                eStart1, eEnd1 = _findBrackets(eqString, cursor, direction=1)
                eStart2, eEnd2 = _findBrackets(eqString, eEnd1, direction=1)
                elem1 = eqString[eStart1:eEnd1]
                elem2 = eqString[eStart2:eEnd2]

                beforeBrace = eqString[0:cursor]
                afterBrace = eqString[eEnd2:]

                eqString = beforeBrace + braceElem + elem1 + "^" + elem2 + afterBrace
            except ValueError:
                return eqString
        return eqString

    for braceKey, braceElem in braceDict.items():
        eqString = replaceBrace(eqString, braceKey, braceElem)
    return eqString
