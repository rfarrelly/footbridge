import math


def get_no_vig_odds_multiway(odds: list):
    """
    :param odds: List of original odds for a multi-way market.
    :return: Tuple of no-vig (fair) odds calculated using the iterative method.
    """
    c, target_overround, accuracy, current_error = 1, 0, 3, 1000
    max_error = (10 ** (-accuracy)) / 2

    fair_odds = list()
    while current_error > max_error:
        f = -1 - target_overround
        for o in odds:
            f += (1 / o) ** c

        f_dash = 0
        for o in odds:
            f_dash += ((1 / o) ** c) * (-math.log(o))

        h = -f / f_dash
        c = c + h

        t = 0
        for o in odds:
            t += (1 / o) ** c
        current_error = abs(t - 1 - target_overround)

        fair_odds = list()
        for o in odds:
            fair_odds.append(round(o**c, 3))

    return tuple(fair_odds)
