# -*- coding: utf-8 -*-

BRACKET_PROCESS = {
    "\\\\*left\s*\\(": "(",
    "\\\\*left\s*\\{": "{",
    "\\\\*right\s*\\)": ")",
    "\\\\*right\s*\\}": "}",
}

SYMBOL_REGULARIZATION = {
    "\\aleph": "ℵ",
    "\\beth": "ℶ",
    "\\daleth": "ℸ",
    "\\gimel": "ℷ",
    "\\alpha": "𝛼",
    "\\iota": "𝜄",
    "\\sigma": "𝜎",
    "\\beta": "𝛽",
    "\\kappa": "𝜅",
    "\\tau": "𝜏",
    "\\gamma": "𝛾",
    "\\lambda": "𝜆",
    "\\upsilon": "𝜐",
    "\\delta": "𝛿",
    "\\mu": "𝜇",
    "\\phi": "𝜙",
    "\\epsilon": "𝜖",
    "\\nu": "𝜈",
    "\\chi": "𝜒",
    "\\zeta": "𝜁",
    "\\xi": "𝜉",
    "\\psi": "𝜓",
    "\\eta": "𝜂",
    "\\pi": "𝜋",
    "\\omega": "𝜔",
    "\\theta": "𝜃",
    "\\rho": "𝜌",
    "\\varepsilon": "𝜀",
    "\\varpi": "𝜛",
    "\\varsigma": "𝜍",
    "\\vartheta": "𝜗",
    "\\varrho": "𝜚",
    "\\varphi": "𝜑",
    "\\digamma": "ϝ",
    "\\varkappa": "𝜘",
    "\\Gamma": "Γ",
    "\\Xi": "Ξ",
    "\\Phi": "Φ",
    "\\Delta": "Δ",
    "\\Pi": "Π",
    "\\Psi": "Ψ",
    "\\Theta": "Θ",
    "\\Sigma": "Σ",
    "\\Omega": "Ω",
    "\\Lambda": "Λ",
    "\\Upsilon": "Υ",
    "\\varGamma": "𝛤",
    "\\varXi": "𝛯",
    "\\varPhi": "𝛷",
    "\\varDelta": "𝛥",
    "\\varPi": "𝛱",
    "\\varPsi": "𝛹",
    "\\varTheta": "𝛩",
    "\\varSigma": "𝛴",
    "\\varOmega": "𝛺",
    "\\varLambda": "𝛬",
    "\\varUpsilon": "𝛶",
    "\\ll": "≪",
    "\\gg": "≫",
    "\\sim": "∼",
    "\\approx": "≈",
    "\\simeq": "≃",
    "\\cong": "≅",
    "\\subseteq": "⊆",
    "\\supseteq": "⊇",
    "\\sqsubseteq": "⊑",
    "\\sqsupseteq": "⊒",
    "\\smile": "⌣",
    "\\frown": "⌢",
    "\\perp": "⟂",
    "\\models": "⊧",
    "\\mid": "∣",
    "\\parallel": "∥",
    "\\vdash": "⊢",
    "\\dashv": "⊣",
    "\\asymp": "≍",
    "\\bowtie": "⋈",
    "\\sqsubset": "⊏",
    "\\sqsupset": "⊐",
    "\\Join": "⨝",
    "\\leqq": "≦",
    "\\geqq": "≧",
    "\\leqslant": "⩽",
    "\\geqslant": "⩾",
    "\\eqslantless": "⪕",
    "\\eqslantgtr": "⪖",
    "\\lesssim": "≲",
    "\\gtrsim": "≳",
    "\\lessapprox": "⪅",
    "\\gtrapprox": "⪆",
    "\\approxeq": "≊",
    "\\lessdot": "⋖",
    "\\gtrdot": "⋗",
    "\\lll": "⋘",
    "\\ggg": "⋙",
    "\\lessgtr": "≶",
    "\\gtrless": "≷",
    "\\lesseqgtr": "⋚",
    "\\gtreqless": "⋛",
    "\\lesseqqgtr": "⪋",
    "\\gtreqqless": "⪌",
    "\\doteqdot": "≑",
    "\\eqcirc": "≖",
    "\\circeq": "≗",
    "\\triangleq": "≜",
    "\\risingdotseq": "≓",
    "\\fallingdotseq": "≒",
    "\\backsim": "∽",
    "\\thicksim": "∼",
    "\\backsimeq": "⋍",
    "\\thickapprox": "≈",
    "\\preccurlyeq": "≼",
    "\\succcurlyeq": "≽",
    "\\curlyeqprec": "⋞",
    "\\curlyeqsucc": "⋟",
    "\\precsim": "≾",
    "\\succsim": "≿",
    "\\precapprox": "⪷",
    "\\succapprox": "⪸",
    "\\subseteqq": "⫅",
    "\\supseteqq": "⫆",
    "\\Subset": "⋐",
    "\\Supset": "⋑",
    "\\vartriangleleft": "⊲",
    "\\vartriangleright": "⊳",
    "\\trianglelefteq": "⊴",
    "\\trianglerighteq": "⊵",
    "\\vDash": "⊨",
    "\\Vdash": "⊩",
    "\\Vvdash": "⊪",
    "\\smallsmile": "⌣",
    "\\smallfrown": "⌢",
    "\\shortmid": "∣",
    "\\shortparallel": "∥",
    "\\bumpeq": "≏",
    "\\Bumpeq": "≎",
    "\\between": "≬",
    "\\pitchfork": "⋔",
    "\\backepsilon": "϶",
    "\\blacktriangleleft": "◀",
    "\\blacktriangleright": "▶",
    "\\nless": "≮",
    "\\ngtr": "≯",
    "\\nleq": "≰",
    "\\ngeq": "≱",
    "\\lneq": "⪇",
    "\\gneq": "⪈",
    "\\lneqq": "≨",
    "\\gneqq": "≩",
    "\\lvertneqq": "≨",
    "\\gvertneqq": "≩",
    "\\lnsim": "⋦",
    "\\gnsim": "⋧",
    "\\lnapprox": "⪉",
    "\\gnapprox": "⪊",
    "\\nprec": "⊀",
    "\\nsucc": "⊁",
    "\\precneqq": "⪵",
    "\\succneqq": "⪶",
    "\\precnsim": "⋨",
    "\\succnsim": "⋩",
    "\\precnapprox": "⪹",
    "\\succnapprox": "⪺",
    "\\nsim": "≁",
    "\\ncong": "≇",
    "\\nshortmid": "∤",
    "\\nshortparallel": "∦",
    "\\nmid": "∤",
    "\\nparallel": "∦",
    "\\nvdash": "⊬",
    "\\nvDash": "⊭",
    "\\nVdash": "⊮",
    "\\nVDash": "⊯",
    "\\ntrianglelefteq": "⋬",
    "\\ntrianglerighteq": "⋭",
    "\\nsubseteq": "⊈",
    "\\nsupseteq": "⊉",
    "\\subsetneq": "⊊",
    "\\supsetneq": "⊋",
    "\\varsubsetneq": "⊊",
    "\\varsupsetneq": "⊋",
    "\\subsetneqq": "⫋",
    "\\supsetneqq": "⫌",
    "\\varsubsetneqq": "⫋",
    "\\varsupsetneqq": "⫌",
    "\\pm": "±",
    "\\mp": "∓",
    "\\ddots": "⋱",
    "\\ldots": "…",
    "\\cdots": "⋯",
    "\\cdot": "⋅",
    "\\circ": "◦",
    "\\bigcirc": "○",
    "\\bmod": "mod",
    "\\sqcap": "⊓",
    "\\sqcup": "⊔",
    "\\wedge": "∧",
    "\\land": "∧",
    "\\vee": "∨",
    "\\lor": "∨",
    "\\triangleleft": "⊲",
    "\\triangleright": "⊳",
    "\\bigtriangleup": "△",
    "\\bigtriangledown": "▽",
    "\\oplus": "⊕",
    "\\ominus": "⊖",
    "\\otimes": "⊗",
    "\\oslash": "⊘",
    "\\odot": "⊙",
    "\\bullet": "∙",
    "\\dagger": "†",
    "\\ddagger": "‡",
    "\\setminus": "⧵",
    "\\smallsetminus": "∖",
    "\\wr": "≀",
    "\\amalg": "⨿",
    "\\ast": "∗",
    "\\star": "⋆",
    "\\diamond": "⋄",
    "\\lhd": "⊲",
    "\\rhd": "⊳",
    "\\unlhd": "⊴",
    "\\unrhd": "⊵",
    "\\dotplus": "∔",
    "\\centerdot": "·",
    "\\ltimes": "⋉",
    "\\rtimes": "⋊",
    "\\leftthreetimes": "⋋",
    "\\rightthreetimes": "⋌",
    "\\circleddash": "⊝",
    "\\uplus": "⊎",
    "\\barwedge": "⊼",
    "\\curlywedge": "⋏",
    "\\curlyvee": "⋎",
    "\\veebar": "⊻",
    "\\intercal": "⊺",
    "\\doublecap": "⋒",
    "\\Cap": "⋒",
    "\\doublecup": "⋓",
    "\\Cup": "⋓",
    "\\circledast": "⊛",
    "\\circledcirc": "⊚",
    "\\boxminus": "⊟",
    "\\boxtimes": "⊠",
    "\\boxdot": "⊡",
    "\\boxplus": "⊞",
    "\\divideontimes": "⋇",
    "\\vartriangle": "▵",
    "\\And": "ς",
    "\\leftarrow": "←",
    "\\rightarrow": "→",
    "\\to": "→",
    "\\longleftarrow": "⟵",
    "\\longrightarrow": "⟶",
    "\\Leftarrow": "⇐",
    "\\Rightarrow": "⇒",
    "\\Longleftarrow": "⟸",
    "\\Longrightarrow": "⟹",
    "\\leftrightarrow": "↔",
    "\\longleftrightarrow": "⟷",
    "\\Leftrightarrow": "⇔",
    "\\Longleftrightarrow": "⟺",
    "\\uparrow": "↑",
    "\\downarrow": "↓",
    "\\Uparrow": "⇑",
    "\\Downarrow": "⇓",
    "\\updownarrow": "↕",
    "\\Updownarrow": "⇕",
    "\\nearrow": "↗",
    "\\searrow": "↘",
    "\\swarrow": "↙",
    "\\nwarrow": "↖",
    "\\iff": "⟺",
    "\\mapstochar": "",
    "\\mapsto": "↦",
    "\\longmapsto": "⟼",
    "\\hookleftarrow": "↩",
    "\\hookrightarrow": "↪",
    "\\leftharpoonup": "↼",
    "\\rightharpoonup": "⇀",
    "\\leftharpoondown": "↽",
    "\\rightharpoondown": "⇁",
    "\\leadsto": "⇝",
    "\\leftleftarrows": "⇇",
    "\\rightrightarrows": "⇉",
    "\\leftrightarrows": "⇆",
    "\\rightleftarrows": "⇄",
    "\\Lleftarrow": "⇚",
    "\\Rrightarrow": "⇛",
    "\\twoheadleftarrow": "↞",
    "\\twoheadrightarrow": "↠",
    "\\leftarrowtail": "↢",
    "\\rightarrowtail": "↣",
    "\\looparrowleft": "↫",
    "\\looparrowright": "↬",
    "\\upuparrows": "⇈",
    "\\downdownarrows": "⇊",
    "\\upharpoonleft": "↿",
    "\\upharpoonright": "↾",
    "\\downharpoonleft": "⇃",
    "\\downharpoonright": "⇂",
    "\\leftrightsquigarrow": "↭",
    "\\rightsquigarrow": "⇝",
    "\\multimap": "⊸",
    "\\nleftarrow": "↚",
    "\\nrightarrow": "↛",
    "\\nLeftarrow": "⇍",
    "\\nRightarrow": "⇏",
    "\\nleftrightarrow": "↮",
    "\\nLeftrightarrow": "⇎",
    "\\dashleftarrow": "⤎",
    "\\dashrightarrow": "⤏",
    "\\curvearrowleft": "↶",
    "\\curvearrowright": "↷",
    "\\circlearrowleft": "↺",
    "\\circlearrowright": "↻",
    "\\leftrightharpoons": "⇋",
    "\\rightleftharpoons": "⇌",
    "\\Lsh": "↰",
    "\\Rsh": "↱",
    "\\hbar": "ℏ",
    "\\ell": "𝓁",
    "\\imath": "𝚤",
    "\\jmath": "𝚥",
    "\\wp": "℘",
    "\\partial": "𝜕",
    "\\Im": "ℑ",
    "\\Re": "ℜ",
    "\\infty": "∞",
    "\\prime": "′",
    "\\forall": "∀",
    "\\exists": "∃",
    "\\smallint": "∫",
    "\\triangle": "△",
    "\\top": "⊤",
    "\\P": "¶",
    "\\S": "§",
    "\\dag": "†",
    "\\ddag": "‡",
    "\\flat": "♭",
    "\\natural": "♮",
    "\\sharp": "♯",
    "\\angle": "각",
    "\\clubsuit": "♣",
    "\\diamondsuit": "♢",
    "\\heartsuit": "♡",
    "\\spadesuit": "♠",
    "\\surd": "√",
    "\\nabla": "∇",
    "\\pounds": "£",
    "\\neg": "¬",
    "\\lnot": "¬",
    "\\Box": "□",
    "\\Diamond": "◊",
    "\\mho": "℧",
    "\\hslash": "ℏ",
    "\\complement": "∁",
    "\\backprime": "‵",
    "\\nexists": "∄",
    "\\Bbbk": "𝕜",
    "\\diagup": "⟋",
    "\\diagdown": "⟍",
    "\\blacktriangle": "▴",
    "\\blacktriangledown": "▾",
    "\\triangledown": "▿",
    "\\eth": "ð",
    "\\square": "□",
    "\\blacksquare": "■",
    "\\lozenge": "◊",
    "\\blacklozenge": "⧫",
    "\\measuredangle": "∡",
    "\\sphericalangle": "∢",
    "\\circledS": "Ⓢ",
    "\\bigstar": "★",
    "\\Finv": "Ⅎ",
    "\\Game": "⅁",
    "\\lbrack": "[",
    "\\rbrack": "]",
    "\\lbrace": "{",
    "\\rbrace": "}",
    "\\backslash": "∖",
    "\\langle": "⟨",
    "\\rangle": "⟩",
    "\\vert": "|",
    "\\|": "‖",
    "\\Vert": "‖",
    "\\lfloor": "⌊",
    "\\rceil": "⌉",
    "\\ulcorner": "⌜",
    "\\urcorner": "⌝",
    "\\llcorner": "⌞",
    "\\lrcorner": "⌟",
}

SYMBOL_REGULARIZATION2 = {
    "𝛼": "α",
    "𝛽": "β",
    "𝛾": "γ",
    "𝛿": "δ",
    "𝜂": "η",
    "𝜃": "θ",
    "𝜅": "κ",
    "𝜆": "λ",
    "𝜇": "μ",
    "𝜋": "π",
    "𝜌": "ρ",
    "𝜎": "σ",
    "𝜏": "τ",
    "𝜑": "φ",
    "𝜙": "φ",
    "𝜔": "ω",
    "１": "1",
    "３": "3",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    "＠": "@",
    "Ｉ": "I",
    "Ｌ": "L",
    "Ｏ": "O",
    "Ｐ": "P",
    "＃": "#",
    "（": "(",
    "）": ")",
    "＊": "*",
    "－": "-",
    "〓": "=",
    "𝚤": "ı",
    "ƒ": "f",
    "ｇ": "g",
    "ｍ": "m",
    "㎎": "mg",
    "㎏": "kg",
    "㎝": "cm",
    "㎞": "km",
    "㎠": "cm^{2}",
    "㎡": "m^{2}",
    "㎤": "cm^{3}",
    "㎥": "m^{3}",
    "＝": "=",
    "＞": ">",
    "［": "[",
    "］": "]",
    "．": ".",
    "：": ":",
    "？": "?",
    "＾": "^",
    "︳": "|",
    "∣": "|",
    "‖": "||",
    "∥": "//",
    "": "//",
    "": "/",
    "∕": "/",
    "∖": "\\",
    "": ".",
    "": "-",
    "—": "-",
    "―": "-",
    "": "□",
    "": "α",
    "六": "六",
    "〉": ">",
    "◯": "○",
    "⌾": "◎",
    "⦾": "◎",
    "➊": "❶",
    "➋": "❷",
    "➌": "❸",
    "➍": "❹",
    "⑴": "(1)",
    "⑵": "(2)",
    "⑶": "(3)",
    "⑷": "(4)",
    "⑸": "(5)",
    "⑹": "(6)",
    "⑺": "(7)",
    "⑻": "(8)",
    "⑼": "(9)",
    "⊏": "ㄷ",
    "㈀": "(ㄱ)",
    "㈁": "(ㄴ)",
    "㈂": "(ㄷ)",
    "㈃": "(ㄹ)",
    "㈄": "(ㅁ)",
    "㈅": "(ㅂ)",
    "㈎": "(가)",
    "㈏": "(나)",
    "㈐": "(다)",
    "㈑": "(라)",
    "㈒": "(마)",
    "㈓": "(바)",
    "㈕": "(아)",
    "｢": "「",
    "｣": "」",
    "…": "...",
    "ʹ": "′",
    "″": "′",
    "․": ".",
    "∙": "·",
    "⋅": "·",
    "ㆍ": "·",
    "⦁": "·",
    "•": "·",
    "‧": "·",
    "◦": "∘",
    "⋆": "*",
    "∗": "*",
    "✱": "*",
    "✴": "*",
    "✻": "*",
    "✽": "*",
    "☓": "X",
    "☐": "□",
    "О": "○",
    "о": "○",
    "⨂": "⊗",
    "☉": "⊙",
    "⊚": "◎",
    "➀": "①",
    "➁": "②",
    "➂": "③",
    "➃": "④",
    "➄": "⑤",
    "Θ": "θ",
    "Ω": "ω",
    "➜": "→",
    "➪": "⇨",
    "⊳": "▷",
    "∆": "△",
    "Δ": "△",
    "▐": "■",
    "▌": "■",
    "Π": "n",
    "╱": "/",
    "│": "|",
    "─": "-",
    "∶": ":",
    "ː": ":",
    "⇢": "→",
    "⟶": "→",
    "⇣": "↓",
}

REX_MAP = {
    "\\/\s*\\/": "평행",
    "\\^\s*\\{\s*C\s*\\}": "여집합",
    "[a-z]′\\([t-z]+\\)|[a-z] \\^ \\{ ′ \\} \\( [t-z]+ \\)": "도함수",
    "[a-z]′\\(\\d+\\)|[a-z] \\^ \\{ ′ \\} \\( \\d+ \\)|[a-z]′\\([a-s]+\\)|[a-z] \\^ \\{ ′ \\} \\( [a-s]+ \\)": "미분계수",
    "[a-z0-9+]!": "계승",
    "<[가-힣]+>": "",
    "[a-z]′′\\([t-z]+\\)|[a-z] \\^ \\{ ′\\s*′ \\} \\( [t-z]+ \\)": "이계도함수",
    "[a-z]′′\\(\\d+\\)|[a-z] \\^ \\{ ′\\s*′ \\} \\( \\d+ \\)|[a-z]′′\\([a-s]+\\)|[a-z] \\^ \\{ ′\\s*′ \\} \\( [a-s]+ \\)": "이차미분계수",
    "[a-z]\\s{0,3}∘\\s{0,3}[a-z]": "합성함수",
    "□[a-zA-Z]{4}": "사각형",
    "△[a-zA-Z]{3}": "삼각형",
    "[a-z]{1}\s*\\/\s*[a-z]{1}[^a-z]": "분수",
    "\\^\s*[k-z]": "지수함수",
    "\\^\s*\\{\s*[^가-힣]*[k-z]+[^가-힣\\}\\{]*\s*\\}": "지수함수",
}


SYMBOL_TO_TEXT = {
    "\\varpropto": "비례",
    "\\div": "나눗셈",
    "\\therefore": "그러므로",
    "\\because": "그러므로",
    "\\neq": "같지 않다",
    "\\ne": "같지 않다",
    "\\notin": "원소가 아니다",
    "\\times": "곱셈",
    "small\\bigcap": "교집합",
    "\\cap": "교집합",
    "\\cup": "합집합",
    "\\propto": "비례",
    "\\bot": "수직",
    "\\equiv": "합동",
    "\\doteq": "대략",
    "\\subset": "부분집합",
    "\\supset": "부분집합",
    "\\in": "원소",
    "\\leq": "부등식",
    "\\le": "부등식",
    "\\geq": "부등식",
    "\\ge": "부등식",
    "\\emptyset": "공집합",
    "\\varnothing": "공집합",
    "\\frac": "분수",
    "\\hat{p}": "표본비율",
    "\\hat": "호",
    "\\hline": "분수",
    "\\int": "적분",
    "\\lim": "극한",
    "\\ln": "자연로그",
    "\\log": "로그",
    "\\max": "최댓값",
    "\\min": "최솟값",
    "\\overleftarrow": "직선",
    "\\overleftrightarrow": "직선",
    "\\prod": "중복순열",
    "\\overline": "변",
    "\\dot": "순환소수",
    "\\underline": "밑줄",
    "\\vec": "벡터",
    "％": "비율",
    "＜": "부등식",
    "～": "범위",
    "＋": "덧셈",
    "": "비율",
    "〈": "부등식",
    "˚": "각도",
    "ﾟ": "각도",
    "℃": "온도",
    "℉": "온도",
    "∼": "범위",
    "⋂": "교집합",
    "⋃": "합집합",
    "≦": "부등식",
    "≧": "부등식",
    "⟂": "수직",
    "⟨": "부등식",
    "⟩": "부등식",
    "²": "^{2}",
    "³": "^{3}",
    "≺": "부등식",
    "≻": "부등식",
    "〈": "부등식",
    "〉": "부등식",
    "~": "범위",
    "°": "각도",
    "%": "비율",
    "<": "부등식",
    ">": "부등식",
    "+": "덧셈",
    "×": "곱셈",
    "÷": "나눗셈",
    "≠": "같지 않다",
    "≤": "작거나 같다",
    "≥": "크거나 같다",
    "∾": "닮음",
    "∴": "그러므로",
    "∵": "이유",
    "∠": "각",
    "⊥": "수직",
    "≡": "합동",
    "≐": "대략",
    "∝": "비례",
    "∈": "원소",
    "⊂": "부분집합",
    "⊃": "부분집합",
    "⊄": "부분집합이 아니다",
    "∉": "원소가 아니다",
    "∪": "합집합",
    "∩": "교집합",
    "∅": "공집합",
    "sin": "심긱힘수",
    "cos": "삼각함수",
    "tan": "삼각함수",
    "sec": "삼각함수",
}


TOTAL_TYPESETS = {
    **BRACKET_PROCESS,
    **SYMBOL_REGULARIZATION,
    **SYMBOL_REGULARIZATION2,
    **REX_MAP,
    **SYMBOL_TO_TEXT,
}
