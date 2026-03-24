"""Score candidate screening-question answers against preferred/ideal values."""


def score_answers(questions, answers):
    """Return a 0.0-1.0 score for screening answers.

    Parameters
    ----------
    questions : list[dict]
        Each dict has at least ``id``, ``type``, and optional preferred fields.
    answers : dict
        Mapping of question id -> candidate's answer (string).

    Returns
    -------
    float
        Aggregate score (average of scoreable questions). 0.0 if none scoreable.
    """
    if not questions or not answers:
        return 0.0

    total = 0.0
    scoreable = 0

    for q in questions:
        qid = q.get('id')
        qtype = q.get('type', 'text')
        answer = answers.get(qid)

        if answer is None:
            continue

        if qtype == 'text':
            # Text questions are not scored
            continue

        if qtype == 'yes_no':
            preferred = q.get('preferred_answer')
            if preferred is not None:
                scoreable += 1
                total += 1.0 if str(answer).strip().lower() == str(preferred).strip().lower() else 0.0

        elif qtype == 'multiple_choice':
            preferred = q.get('preferred_options') or []
            if preferred:
                scoreable += 1
                preferred_lower = [str(p).strip().lower() for p in preferred]
                total += 1.0 if str(answer).strip().lower() in preferred_lower else 0.0

        elif qtype == 'scale':
            preferred_range = q.get('preferred_range')
            if preferred_range and len(preferred_range) == 2:
                scoreable += 1
                try:
                    val = float(answer)
                    lo, hi = float(preferred_range[0]), float(preferred_range[1])
                    scale_min = float(q.get('scale_min', 1))
                    scale_max = float(q.get('scale_max', 10))

                    if lo <= val <= hi:
                        total += 1.0
                    else:
                        # Linear decay based on distance from preferred range
                        dist = min(abs(val - lo), abs(val - hi))
                        span = scale_max - scale_min
                        if span > 0:
                            total += max(0.0, 1.0 - dist / span)
                except (ValueError, TypeError):
                    pass  # un-parseable answer scores 0

    return total / scoreable if scoreable > 0 else 0.0
