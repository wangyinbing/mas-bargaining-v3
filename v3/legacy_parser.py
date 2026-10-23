"""
Legacy parser — verbatim port of the Stage-1 v2 parser (2026-09-15).

Purpose: keep the OLD measurement instrument available so that the new study can
quantify how much of the previously reported "agreement rate" was an artifact of
this parser rather than of agent behaviour.

Do not "improve" this file. It is a measurement artifact by design.
"""

import re


def extract_proposal(text):
    """Port of BargainingGame._extract_proposal (v2)."""
    if text is None:
        return None
    text = text.lower()

    m = re.search(r'[\[({](\d+)\s*[,，]\s*(\d+)[\]})]', text)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    m = re.search(r'(\d+)\s*[:：]\s*(\d+)', text)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    m = re.search(r'(?:我|我拿|我要|me)[^\d]*?(\d+)[^\d]*?(?:你|你拿|你要|you)[^\d]*?(\d+)', text)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    has_intent_en = re.search(r'\b(propose|proposes|offer|offers|suggest|suggests|proposal)\b', text)
    has_intent_cn = re.search(r'(建议|提议|我拿|我要|我提议)', text)
    if has_intent_en or has_intent_cn:
        nums = re.findall(r'(?<!\d)(\d+)(?!\d)', text)
        if len(nums) >= 2:
            return (int(nums[0]), int(nums[1]))
    return None


def expresses_acceptance(text, x, y):
    """Port of BargainingGame._expresses_acceptance (v2, word-boundary version)."""
    if text is None:
        return False
    t = text.lower()

    rejection_en = ["reject", "no", "unacceptable", "unfair", "too little", "too much",
                    "not fair", "cannot", "can not", "decline", "refuse", "disagree"]
    rejection_cn = ["拒绝", "不同意", "不接受", "不行", "不好", "不可以", "不公平", "太少", "太多"]
    for w in rejection_en:
        if re.search(r'\b' + re.escape(w).replace(' ', r'\s+') + r'\b', t):
            return False
    for w in rejection_cn:
        if w in t:
            return False

    acceptance_en = ["accept", "accepted", "deal", "agree", "agreed", "ok", "okay", "yes",
                     "works for me", "sounds good", "fine with me"]
    acceptance_cn = ["同意", "接受", "成交", "可以", "没问题", "好的", "行吧", "就这么"]
    for w in acceptance_en:
        if re.search(r'\b' + re.escape(w).replace(' ', r'\s+') + r'\b', t):
            return True
    for w in acceptance_cn:
        if w in t:
            return True

    return bool(re.search(r'(?<!\d)%d(?!\d)' % x, t)) and bool(re.search(r'(?<!\d)%d(?!\d)' % y, t))


def legacy_free_text_agreement(msg_a, msg_b):
    """Port of check_agreement_free_text."""
    proposal = extract_proposal(msg_a)
    if proposal is None:
        proposal = extract_proposal(msg_b)
    if proposal is None:
        return None
    x, y = proposal
    if expresses_acceptance(msg_a, x, y) and expresses_acceptance(msg_b, x, y):
        return (x, y)
    return None
