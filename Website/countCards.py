"""Hi-Lo card counting and count-based deviations from basic strategy."""

BASIC = 'Play your hand according to basic strategy.'


def countCards(card, count):
    """Hi-Lo: 2-6 count +1, 7-9 count 0, 10/J/Q/K/A count -1."""
    if card in [2, 3, 4, 5, 6]:
        count += 1
    elif card in [10, 'J', 'Q', 'K', 'A']:
        count -= 1
    else:
        count += 0
    return count


def resetCount():
    global count
    count = 0
    print('Count reset to 0 and cards dealt cleared.')


def _deviation(count, playerValue, DealerCard, playerHand):
    """Your original rules. Returns a play, or None if the count changes nothing."""
    if count >= 0:
        if playerValue == 16 and DealerCard == 10:
            return 'Stand'
        if playerValue == 12 and DealerCard == 4:
            return 'Stand'
        if count >= 1:
            if playerValue == 11 and DealerCard == 'A':
                return 'Double'
            if playerValue == 9 and DealerCard == 2:
                return 'Double'
            if count >= 2:
                if playerValue == 12 and DealerCard == 3:
                    return 'Stand'
                if count >= 3:
                    if playerValue == 9 and DealerCard == 7:
                        return 'Double'
                    if playerValue == 12 and DealerCard == 2:
                        return 'Stand'
                    if count >= 4:
                        if playerValue == 10 and DealerCard == 'A':
                            return 'Double'
                        if playerValue == 10 and DealerCard == 10:
                            return 'Double'
                        if playerValue == 15 and DealerCard == 10:
                            return 'Stand'
                        if playerHand == [10, 10] and DealerCard == 6:
                            return 'Split'
                        if count >= 5:
                            if playerValue == 16 and DealerCard == 9:
                                return 'Stand'
                            if playerHand == [10, 10] and DealerCard == 5:
                                return 'Split'
    if count <= -1:
        if playerValue == 13 and DealerCard == 2:
            return 'Stand'
        if playerValue == 12 and DealerCard == 6:
            return 'Stand'
        if count <= -2:
            if playerValue == 12 and DealerCard == 5:
                return 'Stand'
            if playerValue == 13 and DealerCard == 3:
                return 'Stand'
    return None


def action(count, playerValue, DealerCard, playerHand, soft=False):
    """Count-based play, or BASIC if the count doesn't change basic strategy.

    DealerCard is an int (2-10) or 'A'; playerHand is a list of ints and 'A'.
    These rules are for hard totals, so soft hands (an ace counted as 11) are skipped.
    """
    if soft:
        return BASIC
    play = _deviation(count, playerValue, DealerCard, playerHand)
    if play == 'Double' and len(playerHand) != 2:
        return BASIC  # you can only double on your first two cards
    return play or BASIC
