EMOTION_TEMPLATES_CN = [
    "这让他感到很[CONCEPT]。",
    "他现在的心情很[CONCEPT]。",
    "听到这个消息，他变得很[CONCEPT]。",
    "这是一个很[CONCEPT]的时刻。",
    "他的表情看起来很[CONCEPT]。",
    "一种很[CONCEPT]的情绪涌上心头。",
    "整个房间的气氛很[CONCEPT]。",
    "这个故事的结局令人[CONCEPT]。",
    "他陷入了[CONCEPT]的情绪中。",
    "他是一个容易[CONCEPT]的人。"
]

SPATIAL_TEMPLATES_CN = [
    "电梯正在[CONCEPT]移动。",
    "他[CONCEPT]看了一眼。",
    "请把页面[CONCEPT]滚动。",
    "他把手指[CONCEPT]指。",
    "屏幕上的箭头[CONCEPT]移动。",
    "那个物体在[CONCEPT]移动。",
    "请[CONCEPT]看。",
    "价格在[CONCEPT]波动。",
    "他将视线[CONCEPT]移动。",
    "那个标志是[CONCEPT]指的。"
]

EMOTION_TEMPLATES_EN = [
    "This made him feel very [CONCEPT].",
    "His current mood is very [CONCEPT].",
    "Hearing this news, he became very [CONCEPT].",
    "This is a very [CONCEPT] moment.",
    "His expression looks very [CONCEPT].",
    "A very [CONCEPT] emotion welled up.",
    "The atmosphere in the whole room is very [CONCEPT].",
    "The ending of this story is [CONCEPT].",
    "He fell into a [CONCEPT] emotion.",
    "He is a person who easily gets [CONCEPT]."
]


SPATIAL_TEMPLATES_EN = [
    "The elevator is moving [CONCEPT].",
    "He glanced [CONCEPT].",
    "Please scroll the page [CONCEPT].",
    "He pointed his finger [CONCEPT].",
    "The arrow on the screen moved [CONCEPT].",
    "That object is moving [CONCEPT].",
    "Please look [CONCEPT].",
    "The price is fluctuating [CONCEPT].",
    "He moved his gaze [CONCEPT].",
    "That sign is pointing [CONCEPT]."
]


EMOTION_POSITIVE_KEWORDS_CN = ["高兴"]  # ["快乐", "高兴", "兴奋", "满足", "愉快", "喜悦", "幸福", "激动", "乐观", "欢欣"]
EMOTION_NEGATIVE_KEWORDS_CN = ["悲伤"]  # ["悲伤", "沮丧", "失望", "痛苦", "忧郁", "愤怒", "焦虑", "恐惧", "绝望", "烦恼"]
ORIENTATION_POSTTIVE_KEYWORDS_CN = ["向上"]  # ["上升", "提升", "上涨", "升高", "提高", "攀升", "增长", "飞升", "跃升", "飙升"]
ORIENTATION_NEGATIVE_KEYWORDS_CN = ["向下"]  # ["向上", "向下"]

EMOTION_POSITIVE_KEWORDS_EN = ["happy", "joyful", "excited", "pleased", "delighted", "cheerful", "optimistic", "elated", "thrilled"]
EMOTION_NEGATIVE_KEWORDS_EN = ["sad", "unhappy", "miserable", "depressed", "gloomy", "anxious", "worried", "upset", "lonely", "pessimistic", "frustrated"]
ORIENTATION_POSTTIVE_KEYWORDS_EN = ["upward", "up", "front", "forward"]
ORIENTATION_NEGATIVE_KEYWORDS_EN = ["downward", "down", "back", "backward"]


PROMPT_FILTER_KEYWORD_UP_EN = """
Given a sentence with "up", determine its category by these rules:
Category 1 (Physical): "Up" denotes the spatial meaning of "upward" in the physical world. It must refer to a **visible, dynamic and concrete action of moving "upward"** from a person or object.
Category 2 (Abstract): "Up" acts as a spatial metaphor. It represents an abstract meaning of being or going "upward" of an **invisible concept** like number, quantity, emotion or time, etc. (e.g., He is feeling up / The price went up / up to 9 pages / ...).
Category 3 (Idiomatic): "Up" has no "upward" meaning whatsoever. It has no relationship with spatial orientation. It simply pairs with a verb **without** any directional or practical significance (e.g., give up, sum up, show up, shut up, ...).
Please analyze the sentence according to the rules above, and directly output the category ID (1, 2, or 3).
Sentence: {sentence}
Category ID:
"""

PROMPT_FILTER_KEYWORD_DOWN_EN = """
Given a sentence with "down", determine its category by these rules:
Category 1 (Physical): "Down" denotes the spatial meaning of "downward" in the physical world. It must refer to a **visible, dynamic and concrete action of moving "downward"** from a person or object.
Category 2 (Abstract): "Down" acts as a spatial metaphor. It represents an abstract meaning of being or going "downward" of an **invisible concept** like number, quantity, emotion or time, etc. (e.g., He is feeling down / The price went down / ...).
Category 3 (Idiomatic): "Down" has no "downward" meaning whatsoever. It has no relationship with spatial orientation. It simply pairs with a verb **without** any directional or practical significance (e.g., cut down, break down, shut down, ...).
Please directly output the category ID (1, 2, or 3).
Sentence: {sentence}
Category ID:
"""

PROMPT_FILTER_KEYWORD_HAPPY_EN = """
Given a sentence with "happy", determine its category by these rules:
Category 1: The overall sentiment is positive, aligning with the meaning of "happy" (e.g., She is happy to meet her friends).
Category 2: The sentence contains "happy" but the overall meaning is neutral or negative (e.g., He is not happy).
Please directly output the category ID (1 or 2).
Sentence: {sentence}
Category ID:
"""

PROMPT_FILTER_KEYWORD_SAD_EN = """
Given a sentence with "sad", determine its category by these rules:
Category 1: The overall sentiment is negative, aligning with the meaning of "sad" (e.g., She is sad and frustrated).
Category 2: The sentence contains "sad" but the overall meaning is neutral or positive (e.g., He is not sad any more).
Please directly output the category ID (1 or 2).
Sentence: {sentence}
Category ID:
"""

PROMPT_FILTER_KEYWORD_UP_EN_CN = """
给你一条包含单词"up"的句子, 请你判断其类别, 规则如下:
类别1: "up"表示物理世界中空间意义上的”向上“的方向，必须是可以看见的具体的”向上“的动作或现象，例如fly up / jump up / stand up / ...等;
类别2: "up"作为空间隐喻，即表示不可观测的抽象意义上的”向上“，例如He is feeling up / The price went up / ...等;
类别3: "up"没有任何”向上“的含义，只是单纯地和动词搭配，没有任何方向性和实际意义，例如give up / sum up / show up / ... 等。
请你在分析过后直接输出该句的类别序号。
句子: {}
类别:
"""

PROMPT_FILTER_KEYWORD_HAPPY_EN_CN = """
给你一条包含单词"happy"的句子, 请你判断其类别, 规则如下:
类别1: 该句上下文所传达的情感是积极的情绪，与"happy"的意义相符，如She is happy to meet her friends;
类别2: 该句仅仅含有单纯"happy"，但整体表达的语义是中立或消极的，如He is not happy;
请你在分析过后直接输出该句的类别序号。
句子: {}
类别:
"""
