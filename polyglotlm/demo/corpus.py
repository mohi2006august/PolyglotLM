"""Synthetic parallel corpora for the CPU smoke recipe and the test suite.

Every frame is one sentence pattern with aligned slot values in English, Hindi, Tamil, Urdu and
Bengali, so the generator yields both monolingual text and sentence-aligned translations.

These sentences were written for plumbing tests only. They are simple, repetitive and have not
been reviewed by native speakers; do not treat anything trained on them as a language model of
these languages.
"""

from __future__ import annotations

import itertools
import random
import unicodedata
from dataclasses import dataclass

DEMO_LANGS: tuple[str, ...] = ("en", "hi", "ta", "ur", "bn")
DEMO_LICENCE = "CC0-1.0"

Slots = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Frame:
    patterns: dict[str, str]
    slots: dict[str, Slots]

    def shape(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.slots["en"])

    def render(self, lang: str, choice: tuple[int, ...]) -> str:
        values = [self.slots[lang][i][j] for i, j in enumerate(choice)]
        return self.patterns[lang].format(*values)


FRAMES: tuple[Frame, ...] = (
    Frame(  # a crowd at a place
        patterns={
            "en": "There is a big crowd at the {0} today.",
            "hi": "आज {0} में बहुत भीड़ है।",
            "ta": "இன்று {0} கூட்டம் அதிகமாக உள்ளது.",
            "ur": "آج {0} میں بہت بھیڑ ہے۔",
            "bn": "আজ {0} খুব ভিড়।",
        },
        slots={
            "en": (("market", "school", "hospital", "village", "city", "office", "library"),),
            "hi": (("बाज़ार", "स्कूल", "अस्पताल", "गाँव", "शहर", "दफ़्तर", "पुस्तकालय"),),
            "ta": (
                (
                    "சந்தையில்",
                    "பள்ளியில்",
                    "மருத்துவமனையில்",
                    "கிராமத்தில்",
                    "நகரத்தில்",
                    "அலுவலகத்தில்",
                    "நூலகத்தில்",
                ),
            ),
            "ur": (("بازار", "اسکول", "ہسپتال", "گاؤں", "شہر", "دفتر", "کتب خانے"),),
            "bn": (("বাজারে", "স্কুলে", "হাসপাতালে", "গ্রামে", "শহরে", "অফিসে", "গ্রন্থাগারে"),),
        },
    ),
    Frame(  # people travelling tomorrow
        patterns={
            "en": "The {0} will go to the {1} tomorrow.",
            "hi": "{0} कल {1} जाएँगे।",
            "ta": "{0} நாளை {1} செல்வார்கள்.",
            "ur": "{0} کل {1} جائیں گے۔",
            "bn": "{0} আগামীকাল {1} যাবেন।",
        },
        slots={
            "en": (
                ("farmers", "teachers", "doctors", "neighbours"),
                ("market", "school", "hospital", "village", "city", "bank"),
            ),
            "hi": (
                ("किसान", "शिक्षक", "डॉक्टर", "पड़ोसी"),
                ("बाज़ार", "स्कूल", "अस्पताल", "गाँव", "शहर", "बैंक"),
            ),
            "ta": (
                ("விவசாயிகள்", "ஆசிரியர்கள்", "மருத்துவர்கள்", "அண்டை வீட்டார்"),
                (
                    "சந்தைக்கு",
                    "பள்ளிக்கு",
                    "மருத்துவமனைக்கு",
                    "கிராமத்திற்கு",
                    "நகரத்திற்கு",
                    "வங்கிக்கு",
                ),
            ),
            "ur": (
                ("کسان", "اساتذہ", "ڈاکٹر", "پڑوسی"),
                ("بازار", "اسکول", "ہسپتال", "گاؤں", "شہر", "بینک"),
            ),
            "bn": (
                ("কৃষকেরা", "শিক্ষকেরা", "ডাক্তারেরা", "প্রতিবেশীরা"),
                ("বাজারে", "স্কুলে", "হাসপাতালে", "গ্রামে", "শহরে", "ব্যাংকে"),
            ),
        },
    ),
    Frame(  # healthy food
        patterns={
            "en": "Eating {0} is good for your health.",
            "hi": "{0} खाना सेहत के लिए अच्छा है।",
            "ta": "{0} சாப்பிடுவது உடல்நலத்திற்கு நல்லது.",
            "ur": "{0} کھانا صحت کے لیے اچھا ہے۔",
            "bn": "{0} খাওয়া স্বাস্থ্যের জন্য ভালো।",
        },
        slots={
            "en": (("fruit", "green vegetables", "lentils", "yoghurt"),),
            "hi": (("फल", "हरी सब्ज़ियाँ", "दाल", "दही"),),
            "ta": (("பழங்கள்", "பச்சைக் காய்கறிகள்", "பருப்பு", "தயிர்"),),
            "ur": (("پھل", "ہری سبزیاں", "دال", "دہی"),),
            "bn": (("ফল", "সবুজ শাকসবজি", "ডাল", "দই"),),
        },
    ),
    Frame(  # harvest by season
        patterns={
            "en": "The {1} harvest is good in {0}.",
            "hi": "{0} में {1} की फ़सल अच्छी होती है।",
            "ta": "{0} {1} நன்றாக விளையும்.",
            "ur": "{0} میں {1} کی فصل اچھی ہوتی ہے۔",
            "bn": "{0} {1} ফলন ভালো হয়।",
        },
        slots={
            "en": (
                ("winter", "summer", "the rainy season"),
                ("wheat", "rice", "maize", "mustard", "sugarcane"),
            ),
            "hi": (("सर्दी", "गर्मी", "बरसात"), ("गेहूँ", "धान", "मक्का", "सरसों", "गन्ने")),
            "ta": (
                ("குளிர்காலத்தில்", "கோடைக்காலத்தில்", "மழைக்காலத்தில்"),
                ("கோதுமை", "நெல்", "மக்காச்சோளம்", "கடுகு", "கரும்பு"),
            ),
            "ur": (("سردی", "گرمی", "برسات"), ("گندم", "دھان", "مکئی", "سرسوں", "گنے")),
            "bn": (
                ("শীতকালে", "গ্রীষ্মকালে", "বর্ষাকালে"),
                ("গমের", "ধানের", "ভুট্টার", "সরিষার", "আখের"),
            ),
        },
    ),
    Frame(  # government schemes
        patterns={
            "en": "The government has started a new scheme for {0}.",
            "hi": "सरकार ने {0} के लिए नई योजना शुरू की है।",
            "ta": "அரசு {0} புதிய திட்டத்தைத் தொடங்கியுள்ளது.",
            "ur": "حکومت نے {0} کے لیے نئی اسکیم شروع کی ہے۔",
            "bn": "সরকার {0} জন্য নতুন প্রকল্প শুরু করেছে।",
        },
        slots={
            "en": (("farmers", "students", "women", "the elderly", "small traders"),),
            "hi": (("किसानों", "छात्रों", "महिलाओं", "बुज़ुर्गों", "छोटे व्यापारियों"),),
            "ta": (
                (
                    "விவசாயிகளுக்கு",
                    "மாணவர்களுக்கு",
                    "பெண்களுக்கு",
                    "முதியோருக்கு",
                    "சிறு வணிகர்களுக்கு",
                ),
            ),
            "ur": (("کسانوں", "طلبہ", "خواتین", "بزرگوں", "چھوٹے تاجروں"),),
            "bn": (("কৃষকদের", "ছাত্রদের", "মহিলাদের", "প্রবীণদের", "ছোট ব্যবসায়ীদের"),),
        },
    ),
    Frame(  # documents for an office visit
        patterns={
            "en": "Please bring your {0} to the {1}.",
            "hi": "कृपया {0} लेकर {1} आइए।",
            "ta": "தயவுசெய்து {0} எடுத்துக்கொண்டு {1} வாருங்கள்.",
            "ur": "براہ کرم {0} لے کر {1} آئیں۔",
            "bn": "অনুগ্রহ করে {0} নিয়ে {1} আসুন।",
        },
        slots={
            "en": (
                ("identity card", "ration card", "bank passbook"),
                ("village council office", "post office", "bank", "district office"),
            ),
            "hi": (
                ("अपना पहचान पत्र", "अपना राशन कार्ड", "अपनी बैंक पासबुक"),
                ("पंचायत कार्यालय", "डाकघर", "बैंक", "ज़िला कार्यालय"),
            ),
            "ta": (
                ("உங்கள் அடையாள அட்டையை", "உங்கள் ரேஷன் அட்டையை", "உங்கள் வங்கிப் புத்தகத்தை"),
                (
                    "ஊராட்சி அலுவலகத்திற்கு",
                    "அஞ்சல் நிலையத்திற்கு",
                    "வங்கிக்கு",
                    "மாவட்ட அலுவலகத்திற்கு",
                ),
            ),
            "ur": (
                ("اپنا شناختی کارڈ", "اپنا راشن کارڈ", "اپنی بینک پاس بک"),
                ("پنچایت دفتر", "ڈاک خانے", "بینک", "ضلعی دفتر"),
            ),
            "bn": (
                ("আপনার পরিচয়পত্র", "আপনার রেশন কার্ড", "আপনার ব্যাংকের পাসবই"),
                ("পঞ্চায়েত অফিসে", "ডাকঘরে", "ব্যাংকে", "জেলা অফিসে"),
            ),
        },
    ),
    Frame(  # weather
        patterns={
            "en": "{0} {1}.",
            "hi": "{0} {1}।",
            "ta": "{0} {1}.",
            "ur": "{0} {1}۔",
            "bn": "{0} {1}।",
        },
        slots={
            "en": (
                ("This morning", "Yesterday evening", "This week"),
                ("it rained heavily", "a cold wind blew", "the weather stayed clear"),
            ),
            "hi": (
                ("आज सुबह", "कल शाम", "इस हफ़्ते"),
                ("तेज़ बारिश हुई", "ठंडी हवा चली", "मौसम साफ़ रहा"),
            ),
            "ta": (
                ("இன்று காலை", "நேற்று மாலை", "இந்த வாரம்"),
                ("பலத்த மழை பெய்தது", "குளிர்ந்த காற்று வீசியது", "வானம் தெளிவாக இருந்தது"),
            ),
            "ur": (
                ("آج صبح", "کل شام", "اس ہفتے"),
                ("تیز بارش ہوئی", "ٹھنڈی ہوا چلی", "موسم صاف رہا"),
            ),
            "bn": (
                ("আজ সকালে", "গতকাল সন্ধ্যায়", "এই সপ্তাহে"),
                ("প্রবল বৃষ্টি হয়েছে", "ঠান্ডা হাওয়া বইছিল", "আকাশ পরিষ্কার ছিল"),
            ),
        },
    ),
    Frame(  # reading
        patterns={
            "en": "I enjoy reading {0}.",
            "hi": "{0} पढ़ना मुझे अच्छा लगता है।",
            "ta": "{0} படிப்பது எனக்குப் பிடிக்கும்.",
            "ur": "{0} پڑھنا مجھے اچھا لگتا ہے۔",
            "bn": "{0} পড়তে আমার ভালো লাগে।",
        },
        slots={
            "en": (("history", "science", "mathematics", "poetry", "stories"),),
            "hi": (("इतिहास", "विज्ञान", "गणित", "कविता", "कहानियाँ"),),
            "ta": (("வரலாறு", "அறிவியல்", "கணிதம்", "கவிதை", "கதைகள்"),),
            "ur": (("تاریخ", "سائنس", "ریاضی", "شاعری", "کہانیاں"),),
            "bn": (("ইতিহাস", "বিজ্ঞান", "গণিত", "কবিতা", "গল্প"),),
        },
    ),
    Frame(  # medicine dosage
        patterns={
            "en": "Take this medicine {0} a day {1}.",
            "hi": "यह दवा दिन में {0} बार {1} लें।",
            "ta": "இந்த மருந்தை ஒரு நாளைக்கு {0} முறை {1} எடுத்துக்கொள்ளுங்கள்.",
            "ur": "یہ دوا دن میں {0} بار {1} لیں۔",
            "bn": "এই ওষুধ দিনে {0} {1} খাবেন।",
        },
        slots={
            "en": (("once", "twice", "three times"), ("after meals", "before meals", "with water")),
            "hi": (("एक", "दो", "तीन"), ("खाने के बाद", "खाने से पहले", "पानी के साथ")),
            "ta": (
                ("ஒரு", "இரண்டு", "மூன்று"),
                ("உணவுக்குப் பிறகு", "உணவுக்கு முன்", "தண்ணீருடன்"),
            ),
            "ur": (("ایک", "دو", "تین"), ("کھانے کے بعد", "کھانے سے پہلے", "پانی کے ساتھ")),
            "bn": (("একবার", "দুবার", "তিনবার"), ("খাওয়ার পরে", "খাওয়ার আগে", "জলের সঙ্গে")),
        },
    ),
    Frame(  # prices
        patterns={
            "en": "The price of {0} has gone {1}.",
            "hi": "{0} का दाम {1} गया है।",
            "ta": "{0} விலை {1}.",
            "ur": "{0} کی قیمت {1} گئی ہے۔",
            "bn": "{0} দাম {1}।",
        },
        slots={
            "en": (("onions", "tomatoes", "rice", "milk", "oil"), ("up", "down")),
            "hi": (("प्याज़", "टमाटर", "चावल", "दूध", "तेल"), ("बढ़", "घट")),
            "ta": (
                ("வெங்காயத்தின்", "தக்காளியின்", "அரிசியின்", "பாலின்", "எண்ணெயின்"),
                ("உயர்ந்துள்ளது", "குறைந்துள்ளது"),
            ),
            "ur": (("پیاز", "ٹماٹر", "چاول", "دودھ", "تیل"), ("بڑھ", "گھٹ")),
            "bn": (("পেঁয়াজের", "টমেটোর", "চালের", "দুধের", "তেলের"), ("বেড়েছে", "কমেছে")),
        },
    ),
    Frame(  # drinking water
        patterns={
            "en": "Clean drinking water is important for every {0}.",
            "hi": "हर {0} के लिए साफ़ पीने का पानी ज़रूरी है।",
            "ta": "ஒவ்வொரு {0} சுத்தமான குடிநீர் அவசியம்.",
            "ur": "ہر {0} کے لیے صاف پینے کا پانی ضروری ہے۔",
            "bn": "প্রতিটি {0} জন্য পরিষ্কার পানীয় জল জরুরি।",
        },
        slots={
            "en": (("village", "family", "school", "child"),),
            "hi": (("गाँव", "परिवार", "स्कूल", "बच्चे"),),
            "ta": (("கிராமத்திற்கும்", "குடும்பத்திற்கும்", "பள்ளிக்கும்", "குழந்தைக்கும்"),),
            "ur": (("گاؤں", "خاندان", "اسکول", "بچے"),),
            "bn": (("গ্রামের", "পরিবারের", "স্কুলের", "শিশুর"),),
        },
    ),
    Frame(  # transport timings
        patterns={
            "en": "The {0} to the city leaves at {1} o'clock.",
            "hi": "शहर जाने वाली {0} {1} बजे निकलती है।",
            "ta": "நகரத்திற்குச் செல்லும் {0} {1} மணிக்குப் புறப்படும்.",
            "ur": "شہر جانے والی {0} {1} بجے نکلتی ہے۔",
            "bn": "শহরগামী {0} {1}টায় ছাড়ে।",
        },
        slots={
            "en": (("bus", "train"), ("six", "eight", "ten")),
            "hi": (("बस", "रेलगाड़ी"), ("छह", "आठ", "दस")),
            "ta": (("பேருந்து", "ரயில்"), ("ஆறு", "எட்டு", "பத்து")),
            "ur": (("بس", "ریل گاڑی"), ("چھ", "آٹھ", "دس")),
            "bn": (("বাস", "ট্রেন"), ("ছ", "আট", "দশ")),
        },
    ),
)


def _choices(frame: Frame) -> list[tuple[int, ...]]:
    return list(itertools.product(*(range(n) for n in frame.shape())))


def all_sentences(lang: str) -> list[str]:
    """Every sentence the frames can produce in `lang`, in a fixed order."""
    return [frame.render(lang, choice) for frame in FRAMES for choice in _choices(frame)]


def parallel_sentences() -> list[dict[str, str]]:
    """Sentence-aligned translations: one dict of {lang: sentence} per slot combination."""
    return [
        {lang: frame.render(lang, choice) for lang in DEMO_LANGS}
        for frame in FRAMES
        for choice in _choices(frame)
    ]


def sample_documents(lang: str, n: int, seed: int) -> list[str]:
    """`n` documents of two to four random sentences each."""
    rng = random.Random(f"{seed}:{lang}")
    sentences = all_sentences(lang)
    return [" ".join(rng.sample(sentences, rng.randint(2, 4))) for _ in range(n)]


def noisy_raw_corpus(lang: str, n: int, seed: int) -> list[str]:
    """Clean documents plus the defects the data pipeline must remove or repair."""
    rng = random.Random(f"noise:{seed}:{lang}")
    docs = sample_documents(lang, n, seed)
    fresh = sample_documents(lang, 2, seed + 7919)  # not near-duplicates of `docs`
    other = "hi" if lang == "en" else "en"
    noise = [
        *rng.sample(docs, max(1, n // 20)),  # exact duplicates
        *(f"{doc}  {doc.split()[0]}" for doc in rng.sample(docs, max(1, n // 40))),  # near dups
        *sample_documents(other, max(1, n // 25), seed + 1),  # wrong language
        *(unicodedata.normalize("NFD", doc) for doc in rng.sample(docs, max(1, n // 40))),
        f"{docs[0]} https://spam.example/a https://spam.example/b www.spam.example",
        f"{fresh[0]} user{seed}@example.org +91 98765 43210",  # PII to redact
        "1234 5678 !!!! ???? ---- 9999",  # symbols and digits
        docs[2].split()[0],  # too short
        " ".join([docs[3].split()[0]] * 12),  # repetition spam
    ]
    merged = docs + noise
    rng.shuffle(merged)
    return merged


def demo_seed_pairs(n: int, seed: int) -> list[dict[str, str]]:
    """English instruction/response pairs the offline demo teacher can translate.

    Both sides are demo sentences, so the pairs are plumbing fixtures, not useful instructions.
    """
    rng = random.Random(f"seeds:{seed}")
    english = all_sentences("en")
    return [
        {"instruction": a, "response": b} for a, b in (rng.sample(english, 2) for _ in range(n))
    ]


def demo_eval_items(lang: str, n: int, seed: int) -> list[dict[str, object]]:
    """Word-order acceptability items: pick the real sentence among three shuffled variants.

    A language model that has learned the language prefers the grammatical order; one that has
    not is near chance (25%). Synthetic smoke-test items, not a benchmark.
    """
    rng = random.Random(f"eval:{seed}:{lang}")
    items = []
    for i, sentence in enumerate(rng.sample(all_sentences(lang), n)):
        words = sentence.split()
        variants: set[str] = set()
        while len(variants) < 3:
            shuffled = rng.sample(words, len(words))
            if shuffled != words:
                variants.add(" ".join(shuffled))
        choices = [sentence, *sorted(variants)]
        rng.shuffle(choices)
        items.append(
            {
                "type": "mc",
                "id": f"{lang}-{i}",
                "prompt": "",
                "choices": choices,
                "answer": choices.index(sentence),
            }
        )
    return items
