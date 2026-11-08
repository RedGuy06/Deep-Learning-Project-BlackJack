"""Joins Hi-Lo counting (countCards) with the basic-strategy tables."""
import countCards
import lookupActionFromTable as lookup

# True: the cards in the photo count toward the count before picking a play
# (they're on the table, so a counter has already seen them).
# False: decide using only the count from earlier saved photos.
DECISION_USES_CURRENT_CARDS = True


def to_card(rank):
    """'A' stays 'A'; '2'..'10' (and J/Q/K) become ints; anything else is None."""
    rank = str(rank).strip().upper()
    if rank == 'A':
        return 'A'
    if rank in ('J', 'Q', 'K'):
        return 10
    if rank.isdigit() and 2 <= int(rank) <= 10:
        return int(rank)
    return None  # e.g. '0' (not a card)


def recommend(player_ranks, dealer_ranks, count=None, tables=None):
    """Totals, basic play, and (if count is given) the count-adjusted play.

    count is the running count from earlier saved photos, or None to skip counting.
    Raises ValueError with a user-readable message if a hand has no cards.
    """
    player = [c for c in map(to_card, player_ranks) if c is not None]
    dealer = [c for c in map(to_card, dealer_ranks) if c is not None]
    if not dealer:
        raise ValueError("Couldn't find a dealer card in the top half of the photo.")
    if not player:
        raise ValueError("Couldn't find any of your cards in the bottom half of the photo.")

    delta = 0
    for card in player + dealer:
        delta = countCards.countCards(card, delta)

    player_total, soft = lookup.hand_value(player)
    dealer_total, _ = lookup.hand_value(dealer)
    upcard = dealer[0]

    adjusted = False
    if player_total > 21:
        basic = action = "Bust"
    else:
        basic = action = lookup.getActionFromLookup(player, upcard, tables=tables)
        if count is not None:
            decision_count = count + delta if DECISION_USES_CURRENT_CARDS else count
            tweak = countCards.action(decision_count, player_total, upcard, player, soft=soft)
            if tweak != countCards.BASIC and not basic.lower().startswith(tweak.lower()):
                action, adjusted = tweak, True

    return {
        "player_total": player_total,
        "dealer_total": dealer_total,
        "basic_action": basic,
        "action": action,
        "count_adjusted": adjusted,
        "cards": {"dealer": [str(c) for c in dealer], "player": [str(c) for c in player]},
        "delta": delta,
        "count_after": None if count is None else count + delta,
    }


if __name__ == "__main__":
    # Same example as your original script: hand [10, 2] vs dealer 2, count of 5.
    print(recommend(["10", "2"], ["2"], count=5))
